from contextlib import asynccontextmanager
from uuid import uuid4

import psycopg2
from fastapi import FastAPI, HTTPException

from api.schemas import (
    EvaluateResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)
from classification.intent_classifier import IntentClassifier
from config import settings
from evaluation.evaluator import PipelineEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from feedback.feedback_store import FeedbackStore
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from ingestion.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.conn = None
    app.state.intent_classifier = IntentClassifier(model=settings.generation_model)
    app.state.embedder = Embedder(model=settings.embedding_model)
    app.state.prompt_builder = PromptBuilder()
    app.state.response_generator = ResponseGenerator(model=settings.generation_model)

    if settings.database_url:
        conn = psycopg2.connect(settings.database_url)
        app.state.conn = conn

        app.state.vector_store = VectorStore(conn)
        try:
            app.state.bm25_retriever = BM25Retriever(conn)
        except Exception:
            app.state.bm25_retriever = None
            
        if app.state.bm25_retriever:
            app.state.hybrid_retriever = HybridRetriever(
                app.state.vector_store,
                app.state.bm25_retriever,
                alpha=settings.hybrid_alpha,
            )
        else:
            app.state.hybrid_retriever = None
        app.state.feedback_store = FeedbackStore(conn)
        app.state.pipeline_evaluator = PipelineEvaluator(
            FaithfulnessEvaluator(model=settings.generation_model),
            RelevanceEvaluator(model=settings.generation_model),
            conn,
        )
    try:
        yield
    finally:
        conn = getattr(app.state, "conn", None)
        if conn:
            conn.close()


app = FastAPI(title="IntelliSupport", lifespan=lifespan)

def _get_conn(app: FastAPI):
    conn = getattr(app.state, "conn", None)
    if conn is None:
        raise HTTPException(503, "Database not configured")
    return conn

import anyio

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    conn = _get_conn(app)
    intent = await app.state.intent_classifier.aclassify(request.query)
    query_id = f"qry_{uuid4().hex[:8]}"
    response_id = f"rsp_{uuid4().hex[:8]}"
    query_embedding = await app.state.embedder.aembed_text(request.query)

    if app.state.hybrid_retriever:
        chunks = await anyio.to_thread.run_sync(
            app.state.hybrid_retriever.retrieve_with_reranking,
            request.query, query_embedding, request.top_k,
        )
    else:
        chunks = await anyio.to_thread.run_sync(
            app.state.vector_store.similarity_search,
            query_embedding, request.top_k,
        )

    if chunks:
        messages = app.state.prompt_builder.build_rag_prompt(request.query, chunks, intent)
    else:
        messages = app.state.prompt_builder.build_clarification_prompt(request.query, intent)
    
    generated = await app.state.response_generator.agenerate(messages)
    retrieved_chunk_ids = [c.chunk_id for c in chunks]

    def _save_data():
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO intellisupport.queries (query_id, raw_query, intent, intent_confidence) VALUES (%s,%s,%s,%s)",
                (query_id, request.query, intent.intent, intent.confidence),
            )
            cur.execute(
                "INSERT INTO intellisupport.responses (response_id, query_id, response_text, retrieved_chunk_ids) VALUES (%s,%s,%s,%s)",
                (response_id, query_id, generated.response_text, retrieved_chunk_ids),
            )
        conn.commit()

    await anyio.to_thread.run_sync(_save_data)
    
    return QueryResponse(
        query_id=query_id, 
        response_id=response_id, 
        response_text=generated.response_text,
        intent=intent.intent, 
        intent_confidence=intent.confidence,
        retrieved_chunk_ids=retrieved_chunk_ids,
        faithfulness_score=None,
        relevance_score=None
    )


@app.post("/evaluate/{response_id}", response_model=EvaluateResponse)
async def evaluate(response_id: str):
    conn = _get_conn(app)
    def _fetch_query_id():
        with conn.cursor() as cur:
            cur.execute("SELECT query_id FROM intellisupport.responses WHERE response_id = %s", (response_id,))
            row = cur.fetchone()
        return row
    
    row = await anyio.to_thread.run_sync(_fetch_query_id)
    if not row:
        raise HTTPException(404, "response_id not found")
    report = await app.state.pipeline_evaluator.aevaluate_response(row[0], response_id)
    return EvaluateResponse(
        response_id=response_id, 
        faithfulness_score=report.faithfulness_score,
        relevance_score=report.relevance_score, 
        combined_score=report.combined_score,
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(request: FeedbackRequest):
    _get_conn(app)
    try:
        fid = await anyio.to_thread.run_sync(
            app.state.feedback_store.store_feedback,
            request.response_id, request.rating, request.comment
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return FeedbackResponse(feedback_id=fid, message="Feedback recorded")


@app.get("/feedback/summary/{response_id}")
async def feedback_summary(response_id: str):
    _get_conn(app)
    return await anyio.to_thread.run_sync(
        app.state.feedback_store.get_feedback_summary,
        response_id
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    conn = getattr(app.state, "conn", None)
    if not conn:
        return HealthResponse(status="ok", db_connected=False, chunks_indexed=0)
    try:
        def _count_chunks():
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM intellisupport.chunks")
                count = cur.fetchone()[0]
            return count
        count = await anyio.to_thread.run_sync(_count_chunks)
        return HealthResponse(status="ok", db_connected=True, chunks_indexed=count)
    except Exception:
        conn.rollback()
        return HealthResponse(status="ok", db_connected=False, chunks_indexed=0)
