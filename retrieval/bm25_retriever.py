try:
    from rank_bm25 import BM25Okapi
except ModuleNotFoundError:
    class BM25Okapi:
        def __init__(self, corpus: list[list[str]]):
            self.corpus = corpus

        def get_scores(self, query_tokens: list[str]) -> list[float]:
            query_set = set(query_tokens)
            return [float(len(query_set & set(document))) for document in self.corpus]

from retrieval.vector_store import RetrievedChunk, clamp_score


def tokenize(text: str) -> list[str]:
    return text.lower().split()


class BM25Retriever:
    def __init__(self, conn):
        self.chunks: list[RetrievedChunk] = []
        self.index: BM25Okapi | None = None
        self.rebuild_index(conn)

    def rebuild_index(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_id, doc_id, content
                FROM intellisupport.chunks
                ORDER BY id
                """
            )
            rows = cur.fetchall()

        self.chunks = [
            RetrievedChunk(
                chunk_id=row[0],
                doc_id=row[1],
                content=row[2],
                score=0.0,
                retrieval_method="bm25",
            )
            for row in rows
        ]
        tokenized_docs = [tokenize(chunk.content) for chunk in self.chunks]
        self.index = BM25Okapi(tokenized_docs) if tokenized_docs else None

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if self.index is None or not self.chunks:
            return []

        scores = self.index.get_scores(tokenize(query))
        scored_indices = sorted(
            range(len(scores)),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        top_indices = scored_indices[:top_k]
        max_score = max((float(scores[index]) for index in top_indices), default=0.0)
        if max_score <= 0:
            return []

        results: list[RetrievedChunk] = []
        for index in top_indices:
            chunk = self.chunks[index]
            normalized = float(scores[index]) / max_score
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=clamp_score(normalized),
                    retrieval_method="bm25",
                )
            )
        return results
