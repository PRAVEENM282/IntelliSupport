from classification.intent_classifier import IntentResult
from retrieval.vector_store import RetrievedChunk


class PromptBuilder:
    def build_rag_prompt(
        self,
        query: str,
        retrieved_chunks: list[RetrievedChunk],
        intent: IntentResult,
    ) -> list[dict]:
        system_prompt = (
            "You are Nexora's AI Support Assistant. Answer only from the provided context. "
            "If the context is insufficient, say \"I don't have information about that\" and ask "
            "one concise clarifying question. Do not invent policies, prices, security behavior, "
            "or product capabilities. The detected intent is "
            f"{intent.intent} with confidence {intent.confidence:.2f}; use that to guide tone."
        )

        context_lines = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(
                f"[{index}] doc_id={chunk.doc_id} chunk_id={chunk.chunk_id}\n{chunk.content}"
            )
        context = "\n\n".join(context_lines)
        user_prompt = (
            "Retrieved context:\n"
            f"{context}\n\n"
            f"Customer query: {query}\n\n"
            "Answer the customer using the context above. Cite the chunk_id values you used."
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def build_clarification_prompt(self, query: str, intent: IntentResult) -> list[dict]:
        return [
            {
                "role": "system",
                "content": (
                    "You are Nexora's AI Support Assistant. No relevant support context was "
                    "retrieved. Politely acknowledge that the available information is insufficient "
                    "and ask one clarifying question. Do not provide unsupported instructions. "
                    f"The detected intent is {intent.intent}."
                ),
            },
            {"role": "user", "content": f"Customer query: {query}"},
        ]

    def estimate_prompt_tokens(self, messages: list[dict]) -> int:
        return int(sum(len(message["content"].split()) * 1.3 for message in messages))
