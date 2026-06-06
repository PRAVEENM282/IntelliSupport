import json

from pydantic import BaseModel

from config import settings
from retrieval.vector_store import RetrievedChunk


class FaithfulnessResult(BaseModel):
    faithfulness_score: float
    total_claims: int
    supported_claims: int
    unsupported_claims: int
    reasoning: str


class FaithfulnessEvaluator:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key or None)

    def _async_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=settings.openai_api_key or None)

    def evaluate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        context = "\n\n".join(
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id}\n{chunk.content}"
            for chunk in retrieved_chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an impartial evaluator of Nexora support responses. First extract "
                    "each factual claim from the response. Then classify each claim as supported "
                    "only if it can be directly verified from the provided context. If the context "
                    "does not address a claim, it is unsupported. Return only JSON in this exact "
                    'shape: {"total_claims": 5, "supported_claims": 4, '
                    '"unsupported_claims": 1, "reasoning": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Generated response:\n{response_text}"
                ),
            },
        ]

        try:
            response = self._client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            total_claims = max(0, int(data.get("total_claims", 0)))
            supported_claims = max(0, int(data.get("supported_claims", 0)))
            unsupported_claims = max(0, int(data.get("unsupported_claims", 0)))
            if total_claims == 0:
                score = 1.0
            else:
                supported_claims = min(supported_claims, total_claims)
                score = supported_claims / total_claims
            return FaithfulnessResult(
                faithfulness_score=max(0.0, min(1.0, score)),
                total_claims=total_claims,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            if not response_text.strip():
                return FaithfulnessResult(
                    faithfulness_score=1.0,
                    total_claims=0,
                    supported_claims=0,
                    unsupported_claims=0,
                    reasoning=f"No response claims to evaluate: {exc}",
                )
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_claims=1,
                supported_claims=0,
                unsupported_claims=1,
                reasoning=f"Faithfulness evaluation failed: {exc}",
            )

    async def aevaluate(
        self,
        response_text: str,
        retrieved_chunks: list[RetrievedChunk],
    ) -> FaithfulnessResult:
        context = "\n\n".join(
            f"chunk_id={chunk.chunk_id} doc_id={chunk.doc_id}\n{chunk.content}"
            for chunk in retrieved_chunks
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an impartial evaluator of Nexora support responses. First extract "
                    "each factual claim from the response. Then classify each claim as supported "
                    "only if it can be directly verified from the provided context. If the context "
                    "does not address a claim, it is unsupported. Return only JSON in this exact "
                    'shape: {"total_claims": 5, "supported_claims": 4, '
                    '"unsupported_claims": 1, "reasoning": "..."}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Retrieved context:\n{context}\n\n"
                    f"Generated response:\n{response_text}"
                ),
            },
        ]

        try:
            response = await self._async_client().chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content or "{}")
            total_claims = max(0, int(data.get("total_claims", 0)))
            supported_claims = max(0, int(data.get("supported_claims", 0)))
            unsupported_claims = max(0, int(data.get("unsupported_claims", 0)))
            if total_claims == 0:
                score = 1.0
            else:
                supported_claims = min(supported_claims, total_claims)
                score = supported_claims / total_claims
            return FaithfulnessResult(
                faithfulness_score=max(0.0, min(1.0, score)),
                total_claims=total_claims,
                supported_claims=supported_claims,
                unsupported_claims=unsupported_claims,
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as exc:
            if not response_text.strip():
                return FaithfulnessResult(
                    faithfulness_score=1.0,
                    total_claims=0,
                    supported_claims=0,
                    unsupported_claims=0,
                    reasoning=f"No response claims to evaluate: {exc}",
                )
            return FaithfulnessResult(
                faithfulness_score=0.0,
                total_claims=1,
                supported_claims=0,
                unsupported_claims=1,
                reasoning=f"Faithfulness evaluation failed: {exc}",
            )
