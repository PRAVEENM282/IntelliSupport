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

    def _async_client(self):
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=settings.openai_api_key or None)

    def generate(self, messages: list[dict]) -> GeneratedResponse:
        response = self._client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        usage = response.usage
        return GeneratedResponse(
            response_text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
        )

    async def agenerate(self, messages: list[dict]) -> GeneratedResponse:
        response = await self._async_client().chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        usage = response.usage
        return GeneratedResponse(
            response_text=response.choices[0].message.content or "",
            model=response.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            total_tokens=usage.total_tokens if usage else 0,
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

    async def agenerate_with_fallback(
        self,
        messages: list[dict],
        fallback_messages: list[dict],
    ) -> GeneratedResponse:
        primary = await self.agenerate(messages)
        should_retry = (
            len(primary.response_text.strip()) < 20
            or "i don't know" in primary.response_text.lower()
        )
        if not should_retry:
            return primary
        fallback = await self.agenerate(fallback_messages)
        if len(fallback.response_text) > len(primary.response_text):
            return fallback
        return primary
