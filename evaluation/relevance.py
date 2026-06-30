import json

from pydantic import BaseModel

from config import settings
from retrieval.bm25_retriever import tokenize
from retrieval.vector_store import RetrievedChunk


class ChunkRelevanceScore(BaseModel):
    chunk_id: str
    score: int
    reason: str


class RelevanceResult(BaseModel):
    relevance_score: float
    chunk_scores: list[ChunkRelevanceScore]
    query: str


class RelevanceEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key or None)



    def evaluate(self, query: str, retrieved_chunks: list[RetrievedChunk]) -> RelevanceResult:
        if not retrieved_chunks:
            return RelevanceResult(relevance_score=0.0, chunk_scores=[], query=query)

        chunks_text = "\n\n".join(
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id}\n{chunk.content}"
            for chunk in retrieved_chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You evaluate retrieval quality for Nexora support. For each chunk, rate "
                    "whether it helps answer the query: 0 = Not relevant, 1 = Partially relevant, "
                    "2 = Highly relevant. Return only JSON with this shape: "
                    '{"chunk_scores": [{"chunk_id": "chunk_doc_001_0", "score": 2, '
                    '"reason": "..."}]}'
                ),
            },
            {"role": "user", "content": f"Query: {query}\n\nRetrieved chunks:\n{chunks_text}"},
        ]

        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            content = content.strip()
            if content.startswith("```"):
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end+1]
            data = json.loads(content)
            raw_scores = data.get("chunk_scores", [])
            scores = []
            for item in raw_scores:
                try:
                    score = int(item.get("score", 0))
                except (ValueError, TypeError):
                    score = 0
                scores.append(
                    ChunkRelevanceScore(
                        chunk_id=str(item.get("chunk_id", "")),
                        score=max(0, min(2, score)),
                        reason=str(item.get("reason", "")),
                    )
                )
        except Exception:
            query_tokens = set(tokenize(query))
            scores = []
            for chunk in retrieved_chunks:
                chunk_tokens = set(tokenize(chunk.content))
                overlap = len(query_tokens & chunk_tokens)
                score = 2 if overlap >= 2 else 1 if overlap == 1 else 0
                scores.append(
                    ChunkRelevanceScore(
                        chunk_id=chunk.chunk_id,
                        score=score,
                        reason="Fallback lexical relevance score.",
                    )
                )

        if not scores:
            normalized = 0.0
        else:
            normalized = sum(item.score for item in scores) / (2 * len(scores))
        return RelevanceResult(
            relevance_score=max(0.0, min(1.0, normalized)),
            chunk_scores=scores,
            query=query,
        )



    def evaluate_batch(
        self,
        queries_and_chunks: list[tuple[str, list[RetrievedChunk]]],
    ) -> list[RelevanceResult]:
        return [self.evaluate(query, chunks) for query, chunks in queries_and_chunks]
