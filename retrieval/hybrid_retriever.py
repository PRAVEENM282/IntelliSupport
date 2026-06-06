from retrieval.bm25_retriever import BM25Retriever, tokenize
from retrieval.vector_store import RetrievedChunk, VectorStore, clamp_score


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_retriever: BM25Retriever,
        alpha: float = 0.7,
    ):
        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.alpha = alpha

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        vector_results = self.vector_store.similarity_search(query_embedding, top_k=top_k * 2)
        bm25_results = self.bm25_retriever.search(query, top_k=top_k * 2)

        by_chunk_id: dict[str, dict[str, RetrievedChunk]] = {}
        for chunk in vector_results:
            by_chunk_id.setdefault(chunk.chunk_id, {})["vector"] = chunk
        for chunk in bm25_results:
            by_chunk_id.setdefault(chunk.chunk_id, {})["bm25"] = chunk

        merged: list[RetrievedChunk] = []
        for methods in by_chunk_id.values():
            vector_chunk = methods.get("vector")
            bm25_chunk = methods.get("bm25")
            base_chunk = vector_chunk or bm25_chunk
            if base_chunk is None:
                continue

            vector_score = vector_chunk.score if vector_chunk else 0.0
            bm25_score = bm25_chunk.score if bm25_chunk else 0.0
            if vector_chunk and bm25_chunk:
                final_score = self.alpha * vector_score + (1 - self.alpha) * bm25_score
                method = "hybrid"
            elif vector_chunk:
                final_score = self.alpha * vector_score
                method = "vector"
            else:
                final_score = (1 - self.alpha) * bm25_score
                method = "bm25"

            merged.append(
                RetrievedChunk(
                    chunk_id=base_chunk.chunk_id,
                    doc_id=base_chunk.doc_id,
                    content=base_chunk.content,
                    score=clamp_score(final_score),
                    retrieval_method=method,
                )
            )

        return sorted(merged, key=lambda chunk: chunk.score, reverse=True)[:top_k]

    def retrieve_with_reranking(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        candidates = self.retrieve(query, query_embedding, top_k=top_k * 3)
        query_tokens = set(tokenize(query))

        reranked: list[RetrievedChunk] = []
        for chunk in candidates:
            chunk_tokens = set(tokenize(chunk.content))
            union = query_tokens | chunk_tokens
            jaccard = len(query_tokens & chunk_tokens) / len(union) if union else 0.0
            rerank_score = 0.8 * chunk.score + 0.2 * jaccard
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    content=chunk.content,
                    score=clamp_score(rerank_score),
                    retrieval_method=chunk.retrieval_method,
                )
            )

        return sorted(reranked, key=lambda chunk: chunk.score, reverse=True)[:top_k]
