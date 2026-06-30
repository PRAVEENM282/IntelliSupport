import json

from pydantic import BaseModel

from config import settings


ALLOWED_INTENTS = {
    "billing": "Questions about pricing, invoices, plans, refunds",
    "technical_issue": "Bugs, errors, crashes, unexpected behavior",
    "feature_request": "Asking about features, capabilities, roadmap",
    "integration": "Questions about third-party integrations",
    "account_management": "Login, permissions, team members, 2FA",
    "data_and_export": "Exports, backups, data management",
    "general_inquiry": "Anything that does not fit the above",
}


class IntentResult(BaseModel):
    intent: str
    confidence: float


def _fuzzy_match_intent(raw: str) -> str:
    """Best-effort match of an LLM-returned intent to ALLOWED_INTENTS keys."""
    normalized = (raw or "").lower().replace(" ", "_").replace("-", "_").strip()
    if normalized in ALLOWED_INTENTS:
        return normalized
    for key in ALLOWED_INTENTS:
        if key in normalized or normalized in key:
            return key
    return "general_inquiry"


class IntentClassifier:
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key or None)



    def classify(self, query: str) -> IntentResult:
        labels = "\n".join(f"- {label}: {description}" for label, description in ALLOWED_INTENTS.items())
        messages = [
            {
                "role": "system",
                "content": (
                    "You classify Nexora customer support queries. Use exactly one of these labels:\n"
                    f"{labels}\n\n"
                    'Respond only as JSON: {"intent": "...", "confidence": 0.0}. '
                    "The confidence must be a float between 0 and 1."
                ),
            },
            {"role": "user", "content": query},
        ]

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
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return IntentResult(intent="general_inquiry", confidence=0.0)
        intent = data.get("intent", "general_inquiry")
        try:
            confidence = float(data.get("confidence", 0.0))
        except (ValueError, TypeError):
            confidence = 0.0
        if intent not in ALLOWED_INTENTS:
            intent = _fuzzy_match_intent(intent)
        return IntentResult(intent=intent, confidence=max(0.0, min(1.0, confidence)))



    def classify_batch(self, queries: list[str]) -> list[IntentResult]:
        return [self.classify(query) for query in queries]
