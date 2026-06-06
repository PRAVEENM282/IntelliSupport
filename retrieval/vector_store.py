from pydantic import BaseModel


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(str(float(value)) for value in values) + "]"


def clamp_score(score: float) -> float:
    return max(0.0, min(1.0, float(score)))


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    score: float
    retrieval_method: str


class VectorStore:
    def __init__(self, conn):
        self.conn = conn

    def similarity_search(self, query_embedding: list[float], top_k: int = 5) -> list[RetrievedChunk]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, content, 1 - (embedding <=> %s::vector) AS score
                FROM intellisupport.chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vector_literal(query_embedding), vector_literal(query_embedding), top_k),
            )
            rows = cur.fetchall()

        return [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=clamp_score(row[3]),
                retrieval_method="vector",
            )
            for row in rows
        ]

    def similarity_search_with_threshold(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.75,
    ) -> list[RetrievedChunk]:
        return [
            chunk
            for chunk in self.similarity_search(query_embedding, top_k=top_k)
            if chunk.score >= threshold
        ]
