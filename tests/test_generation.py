import pytest

from classification.intent_classifier import IntentClassifier
from classification.intent_classifier import IntentResult
from config import settings
from generation.prompt_builder import PromptBuilder
from generation.response_generator import ResponseGenerator
from retrieval.vector_store import RetrievedChunk


def _requires_openai():
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live OpenAI test")


def test_classify_billing_intent():
    _requires_openai()
    result = IntentClassifier().classify("How do I upgrade my subscription plan?")
    assert result.intent == "billing"


def test_classify_technical_intent():
    _requires_openai()
    result = IntentClassifier().classify("The app keeps crashing when I open a project")
    assert result.intent == "technical_issue"


def test_classify_confidence_range():
    _requires_openai()
    result = IntentClassifier().classify("Can Nexora send alerts to Slack?")
    assert 0.0 <= result.confidence <= 1.0


def test_build_rag_prompt_structure():
    messages = PromptBuilder().build_rag_prompt(
        "How do I set up 2FA?",
        [
            RetrievedChunk(
                chunk_id="chunk_doc_008_0",
                doc_id="doc_008",
                content="Enable two-factor authentication from Account Settings > Security.",
                score=1.0,
                retrieval_method="hybrid",
            )
        ],
        IntentResult(intent="account_management", confidence=0.9),
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"


def test_prompt_contains_chunk_ids():
    chunk = RetrievedChunk(
        chunk_id="chunk_doc_008_0",
        doc_id="doc_008",
        content="Enable two-factor authentication from Account Settings > Security.",
        score=1.0,
        retrieval_method="hybrid",
    )
    messages = PromptBuilder().build_rag_prompt(
        "How do I set up 2FA?",
        [chunk],
        IntentResult(intent="account_management", confidence=0.9),
    )
    assert chunk.chunk_id in messages[1]["content"]


def test_generate_response_fields():
    _requires_openai()
    messages = [
        {"role": "system", "content": "You are Nexora's AI Support Assistant."},
        {"role": "user", "content": "Briefly say that Nexora can help with projects."},
    ]
    response = ResponseGenerator(max_tokens=80).generate(messages)
    assert response.response_text
    assert response.total_tokens > 0
