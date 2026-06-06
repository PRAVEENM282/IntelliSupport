import time

from config import settings
from ingestion.chunker import Chunk


class EmbeddingError(Exception):
    pass


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


class Embedder:
    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 100):
        self.model = model
        self.batch_size = batch_size

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key or None)

    def _embed_inputs_with_retry(self, texts: list[str]) -> list[list[float]]:
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                response = self._client().embeddings.create(model=self.model, input=texts)
                ordered = sorted(response.data, key=lambda item: item.index)
                return [list(item.embedding) for item in ordered]
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    break
                time.sleep(2**attempt)
        raise EmbeddingError(f"Embedding API failed after retries: {last_error}")

    def embed_text(self, text: str) -> list[float]:
        return self._embed_inputs_with_retry([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            embeddings.extend(self._embed_inputs_with_retry(batch))
        return embeddings

    def embed_and_store_chunks(self, chunks: list[Chunk], conn) -> int:
        from psycopg2.extras import Json

        if not chunks:
            return 0

        embeddings = self.embed_batch([chunk.content for chunk in chunks])
        rows_written = 0
        with conn.cursor() as cur:
            for chunk, embedding in zip(chunks, embeddings):
                cur.execute(
                    """
                    INSERT INTO intellisupport.chunks
                        (chunk_id, doc_id, content, chunk_index, token_count, embedding, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                    ON CONFLICT (chunk_id)
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        chunk_index = EXCLUDED.chunk_index,
                        token_count = EXCLUDED.token_count,
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.content,
                        chunk.chunk_index,
                        chunk.token_count,
                        _vector_literal(embedding),
                        Json(chunk.metadata),
                    ),
                )
                rows_written += cur.rowcount
        conn.commit()
        return rows_written
