from pydantic import BaseModel

from config import settings


class GeneratedResponse(BaseModel):
    response_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseGenerator:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _client(self):
        from openai import OpenAI

        return OpenAI(api_key=settings.openai_api_key or None)



    def generate(self, messages: list[dict]) -> GeneratedResponse:
        response = self._client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        usage = getattr(response, "usage", None)
        return GeneratedResponse(
            response_text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) if usage else 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) if usage else 0,
            total_tokens=getattr(usage, "total_tokens", 0) if usage else 0,
        )



    def generate_with_fallback(
        self,
        messages: list[dict],
        fallback_messages: list[dict],
    ) -> GeneratedResponse:
        primary = self.generate(messages)
        should_retry = (
            len(primary.response_text.strip()) < 20
            or "i don't know" in primary.response_text.lower()
        )
        if not should_retry:
            return primary
        fallback = self.generate(fallback_messages)
        if len(fallback.response_text) > len(primary.response_text):
            return fallback
        return primary


