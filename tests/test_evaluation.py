from pathlib import Path

import pytest

psycopg2 = pytest.importorskip("psycopg2")

from config import settings
from evaluation.evaluator import PipelineEvaluator
from evaluation.faithfulness import FaithfulnessEvaluator
from evaluation.relevance import RelevanceEvaluator
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader, SEED_DOCUMENTS
from retrieval.vector_store import RetrievedChunk


BENCHMARK_TEST_CASES = [
    {
        "query": "How do I add a new member to my team?",
        "expected_doc_ids": ["doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "What happens if I cancel my subscription?",
        "expected_doc_ids": ["doc_003"],
        "expected_intent": "billing",
    },
    {
        "query": "How do I connect Nexora to Slack?",
        "expected_doc_ids": ["doc_004"],
        "expected_intent": "integration",
    },
    {
        "query": "I forgot my password and can't log in",
        "expected_doc_ids": ["doc_008", "doc_002"],
        "expected_intent": "account_management",
    },
    {
        "query": "How do I export my project data as CSV?",
        "expected_doc_ids": ["doc_007"],
        "expected_intent": "data_and_export",
    },
    {
        "query": "The webhook I set up isn't receiving any events",
        "expected_doc_ids": ["doc_009"],
        "expected_intent": "technical_issue",
    },
    {
        "query": "Can I use custom templates for new projects?",
        "expected_doc_ids": ["doc_005"],
        "expected_intent": "feature_request",
    },
    {
        "query": "How do I enable two-factor authentication for my account?",
        "expected_doc_ids": ["doc_008"],
        "expected_intent": "account_management",
    },
]


@pytest.fixture(scope="module")
def sample_chunks():
    return [
        RetrievedChunk(
            chunk_id="chunk_doc_008_0",
            doc_id="doc_008",
            content=(
                "Nexora users enable two-factor authentication from Account Settings, "
                "then Security, by scanning a QR code and saving recovery codes."
            ),
            score=1.0,
            retrieval_method="hybrid",
        )
    ]


@pytest.fixture(scope="module")
def seeded_conn():
    if not settings.database_url:
        pytest.skip("DATABASE_URL is required for benchmark tests")
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for benchmark tests")
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


@pytest.fixture(scope="module")
def benchmark_report(seeded_conn):
    evaluator = PipelineEvaluator(
        FaithfulnessEvaluator(),
        RelevanceEvaluator(),
        seeded_conn,
    )
    return evaluator.run_benchmark(BENCHMARK_TEST_CASES)


def test_faithfulness_score_range(sample_chunks):
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live faithfulness test")
    result = FaithfulnessEvaluator().evaluate(
        "Enable two-factor authentication from Account Settings > Security and save recovery codes.",
        sample_chunks,
    )
    assert 0.0 <= result.faithfulness_score <= 1.0


def test_relevance_score_range(sample_chunks):
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live relevance test")
    result = RelevanceEvaluator().evaluate(
        "How do I enable two-factor authentication?",
        sample_chunks,
    )
    assert 0.0 <= result.relevance_score <= 1.0


def test_benchmark_hit_rate(benchmark_report):
    assert benchmark_report.retrieval_hit_rate >= 0.6


def test_benchmark_intent_accuracy(benchmark_report):
    assert benchmark_report.intent_accuracy >= 0.75


def test_benchmark_avg_faithfulness(benchmark_report):
    assert benchmark_report.avg_faithfulness >= 0.6
