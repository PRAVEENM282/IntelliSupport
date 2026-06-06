from pydantic import BaseModel

from ingestion.loader import Document


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    token_count: int
    metadata: dict = {}


class DocumentChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_document(self, document: Document) -> list[Chunk]:
        tokens = document.content.split()
        if not tokens:
            return []

        chunks: list[Chunk] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        chunk_index = 0

        while start < len(tokens):
            end = min(start + self.chunk_size, len(tokens))
            chunk_tokens = tokens[start:end]
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{document.doc_id}_{chunk_index}",
                    doc_id=document.doc_id,
                    content=" ".join(chunk_tokens),
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                    metadata={
                        "title": document.title,
                        "source_url": document.source_url,
                    },
                )
            )
            if end == len(tokens):
                break
            start += step
            chunk_index += 1

        return chunks

    def chunk_batch(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
