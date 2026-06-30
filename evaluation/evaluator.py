from pydantic import BaseModel

from classification.intent_classifier import IntentClassifier
from config import settings
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from ingestion.embedder import Embedder
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import RetrievedChunk, VectorStore


class EvaluationReport(BaseModel):
    query_id: str
    response_id: str
    faithfulness_score: float
    relevance_score: float
    combined_score: float


class BenchmarkReport(BaseModel):
    total_cases: int
    avg_faithfulness: float
    avg_relevance: float
    avg_combined: float
    retrieval_hit_rate: float
    intent_accuracy: float


class PipelineEvaluator:
    def __init__(
        self,
        faithfulness_evaluator: FaithfulnessEvaluator,
        relevance_evaluator: RelevanceEvaluator,
        conn,
    ):
        self.faithfulness_evaluator = faithfulness_evaluator
        self.relevance_evaluator = relevance_evaluator
        self.conn = conn

    def evaluate_response(self, query_id: str, response_id: str) -> EvaluationReport:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT raw_query FROM intellisupport.queries WHERE query_id = %s",
                (query_id,),
            )
            query_row = cur.fetchone()
            cur.execute(
                """
                SELECT response_text, retrieved_chunk_ids
                FROM intellisupport.responses
                WHERE response_id = %s AND query_id = %s
                """,
                (response_id, query_id),
            )
            response_row = cur.fetchone()

        if query_row is None or response_row is None:
            raise ValueError("query_id or response_id not found")

        raw_query = query_row[0]
        response_text = response_row[0]
        chunk_ids = response_row[1] or []
        retrieved_chunks = self._load_chunks_by_id(chunk_ids)

        faithfulness = self.faithfulness_evaluator.evaluate(response_text, retrieved_chunks)
        relevance = self.relevance_evaluator.evaluate(raw_query, retrieved_chunks)
        combined = (faithfulness.faithfulness_score + relevance.relevance_score) / 2

        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE intellisupport.responses
                SET faithfulness_score = %s, relevance_score = %s
                WHERE response_id = %s
                """,
                (faithfulness.faithfulness_score, relevance.relevance_score, response_id),
            )
        self.conn.commit()

        return EvaluationReport(
            query_id=query_id,
            response_id=response_id,
            faithfulness_score=faithfulness.faithfulness_score,
            relevance_score=relevance.relevance_score,
            combined_score=combined,
        )



    def run_benchmark(self, test_cases: list[dict]) -> BenchmarkReport:
        classifier = IntentClassifier(model=settings.generation_model)
        embedder = Embedder(model=settings.embedding_model)
        vector_store = VectorStore(self.conn)
        bm25_retriever = BM25Retriever(self.conn)
        hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_retriever=bm25_retriever,
            alpha=settings.hybrid_alpha,
        )
        prompt_builder = PromptBuilder()
        response_generator = ResponseGenerator(model=settings.generation_model)

        faithfulness_scores: list[float] = []
        relevance_scores: list[float] = []
        retrieval_hits = 0
        intent_hits = 0

        for test_case in test_cases:
            query = test_case["query"]
            expected_doc_ids = set(test_case["expected_doc_ids"])
            expected_intent = test_case["expected_intent"]

            intent = classifier.classify(query)
            if intent.intent == expected_intent:
                intent_hits += 1

            query_embedding = embedder.embed_text(query)
            chunks = hybrid_retriever.retrieve_with_reranking(query, query_embedding, top_k=settings.top_k)
            if any(chunk.doc_id in expected_doc_ids for chunk in chunks):
                retrieval_hits += 1

            if chunks:
                messages = prompt_builder.build_rag_prompt(query, chunks, intent)
            else:
                messages = prompt_builder.build_clarification_prompt(query, intent)
            generated = response_generator.generate(messages)

            faithfulness = self.faithfulness_evaluator.evaluate(generated.response_text, chunks)
            relevance = self.relevance_evaluator.evaluate(query, chunks)
            faithfulness_scores.append(faithfulness.faithfulness_score)
            relevance_scores.append(relevance.relevance_score)

        total = len(test_cases)
        avg_faithfulness = sum(faithfulness_scores) / total if total else 0.0
        avg_relevance = sum(relevance_scores) / total if total else 0.0
        avg_combined = (avg_faithfulness + avg_relevance) / 2
        return BenchmarkReport(
            total_cases=total,
            avg_faithfulness=avg_faithfulness,
            avg_relevance=avg_relevance,
            avg_combined=avg_combined,
            retrieval_hit_rate=retrieval_hits / total if total else 0.0,
            intent_accuracy=intent_hits / total if total else 0.0,
        )

    def _load_chunks_by_id(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        if not chunk_ids:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, content
                FROM intellisupport.chunks
                WHERE chunk_id = ANY(%s)
                """,
                (chunk_ids,),
            )
            rows = cur.fetchall()
        by_id = {
            row[0]: RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=1.0,
                retrieval_method="stored",
            )
            for row in rows
        }
        return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]
