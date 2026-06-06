import pytest

from config import settings
from ingestion.chunker import DocumentChunker
from ingestion.embedder import Embedder
from ingestion.loader import DocumentLoader


def test_load_from_dict_valid():
    data = {
        "doc_id": "doc_001",
        "title": "Test Document",
        "content": "This is a valid support document.",
        "source_url": "https://example.com",
        "metadata": {"category": "test"},
    }
    document = DocumentLoader().load_from_dict(data)
    assert document.doc_id == data["doc_id"]
    assert document.title == data["title"]
    assert document.content == data["content"]
    assert document.source_url == data["source_url"]
    assert document.metadata == data["metadata"]


def test_load_from_dict_invalid_doc_id():
    with pytest.raises(ValueError):
        DocumentLoader().load_from_dict(
            {"doc_id": "document_1", "title": "Bad", "content": "Valid content"}
        )


def test_load_from_dict_empty_content():
    with pytest.raises(ValueError):
        DocumentLoader().load_from_dict(
            {"doc_id": "doc_001", "title": "Bad", "content": ""}
        )


def test_chunk_document_chunk_ids():
    document = DocumentLoader().load_from_dict(
        {
            "doc_id": "doc_001",
            "title": "Chunking",
            "content": " ".join(f"word{i}" for i in range(80)),
        }
    )
    chunks = DocumentChunker(chunk_size=20, chunk_overlap=5).chunk_document(document)
    assert chunks
    assert all(chunk.chunk_id.startswith("chunk_doc_001_") for chunk in chunks)


def test_chunk_overlap():
    document = DocumentLoader().load_from_dict(
        {
            "doc_id": "doc_001",
            "title": "Overlap",
            "content": " ".join(f"word{i}" for i in range(25)),
        }
    )
    chunks = DocumentChunker(chunk_size=10, chunk_overlap=3).chunk_document(document)
    first_tokens = chunks[0].content.split()
    second_tokens = chunks[1].content.split()
    assert first_tokens[-3:] == second_tokens[:3]


def test_embed_text_shape():
    if not settings.openai_api_key:
        pytest.skip("OPENAI_API_KEY is required for live embedding test")
    embedding = Embedder().embed_text("Nexora support embedding smoke test.")
    assert len(embedding) == 1536
