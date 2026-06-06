from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from config import settings
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader, SEED_DOCUMENTS
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.vector_store import VectorStore


@pytest.fixture(scope="module")
def seeded_conn():
    if not settings.database_url:
        pytest.skip("DATABASE_URL is required for retrieval tests")
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required to seed embeddings")
    try:
        conn = psycopg2.connect(settings.database_url)
    except Exception as exc:
        pytest.skip(f"Could not connect to database: {exc}")

    migration = Path("database/migrations/001_initial.sql").read_text()
    with conn.cursor() as cur:
        cur.execute(migration)
        cur.execute("SELECT COUNT(*) FROM intellisupport.chunks")
        chunk_count = cur.fetchone()[0]
    conn.commit()

    if chunk_count == 0:
        loader = DocumentLoader()
        documents = loader.load_batch(SEED_DOCUMENTS)
        loader.save_to_db(documents, conn)
        chunks = DocumentChunker().chunk_batch(documents)
        Embedder().embed_and_store_chunks(chunks, conn)

    yield conn
    conn.close()


def test_vector_similarity_search_returns_k_results(seeded_conn):
    embedding = Embedder().embed_text("billing subscription plans")
    results = VectorStore(seeded_conn).similarity_search(embedding, top_k=5)
    assert len(results) == 5


def test_vector_similarity_scores_range(seeded_conn):
    embedding = Embedder().embed_text("billing subscription plans")
    results = VectorStore(seeded_conn).similarity_search(embedding, top_k=5)
    assert all(0.0 <= chunk.score <= 1.0 for chunk in results)


def test_bm25_search_returns_results(seeded_conn):
    results = BM25Retriever(seeded_conn).search("billing subscription", top_k=5)
    assert len(results) >= 1


def test_bm25_keyword_relevance(seeded_conn):
    results = BM25Retriever(seeded_conn).search("two factor authentication", top_k=5)
    assert "doc_008" in [chunk.doc_id for chunk in results[:3]]


def test_hybrid_retriever_score_range(seeded_conn):
    embedding = Embedder().embed_text("export project data as CSV")
    vector_store = VectorStore(seeded_conn)
    bm25 = BM25Retriever(seeded_conn)
    results = HybridRetriever(vector_store, bm25).retrieve(
        "export project data as CSV",
        embedding,
        top_k=5,
    )
    assert all(0.0 <= chunk.score <= 1.0 for chunk in results)


def test_hybrid_deduplication(seeded_conn):
    embedding = Embedder().embed_text("export project data as CSV")
    vector_store = VectorStore(seeded_conn)
    bm25 = BM25Retriever(seeded_conn)
    results = HybridRetriever(vector_store, bm25).retrieve(
        "export project data as CSV",
        embedding,
        top_k=5,
    )
    chunk_ids = [chunk.chunk_id for chunk in results]
    assert len(chunk_ids) == len(set(chunk_ids))
