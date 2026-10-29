from dataclasses import dataclass

from openai import OpenAI

from app.config import Settings

import openai

from collections.abc import Generator
from time import monotonic

from app.llm.errors import (
    LLMAccessError,
    LLMBalanceError,
    LLMClientError,
    LLMConnectionError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRequestError,
    LLMTimeoutError,
)


Message = dict[str, str]


@dataclass(frozen=True)
class LLMResult:
    text: str | None
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str

@dataclass(frozen=True)
class LLMTextDelta:
    text: str


@dataclass(frozen=True)
class LLMStreamCompleted:
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


LLMStreamEvent = LLMTextDelta | LLMStreamCompleted

class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(
            base_url=str(settings.base_url),
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.request_timeout_seconds,
            max_retries=settings.max_retries,

        )

    def generate(self, messages: list[Message]) -> LLMResult:
        try:
            return self._generate_once(messages)
        except openai.APIError as error:
            raise self._convert_error(error) from error

    def stream(
            self,
            messages: list[Message],
    ) -> Generator[LLMStreamEvent, None, None]:
        try:
            yield from self._stream_once(messages)
        except openai.APIError as error:
            raise self._convert_error(error) from error


    def _generate_once(self, messages: list[Message]) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
        )

        if not response.choices:
            raise LLMProviderError("Провайдер вернул ответ без вариантов продолжения.")

        choice = response.choices[0]
        usage = response.usage

        return LLMResult(
            text=choice.message.content,
            model=response.model,
            finish_reason=choice.finish_reason,
            prompt_tokens=usage.prompt_tokens if usage is not None else None,
            completion_tokens=usage.completion_tokens if usage is not None else None,
            total_tokens=usage.total_tokens if usage is not None else None,
            response_id=response.id,
        )

    def _stream_once(
            self,
            messages: list[Message],
    ) -> Generator[LLMStreamEvent, None, None]:
        stream = self._client.chat.completions.create(
            model=self._settings.model,
            messages=messages,
            temperature=self._settings.temperature,
            max_completion_tokens=self._settings.max_output_tokens,
            stream=True,
            stream_options={"include_usage": True},
        )

        started_at = monotonic()
        model: str | None = None
        finish_reason: str | None = None
        response_id: str | None = None
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        total_tokens: int | None = None

        try:
            for chunk in stream:
                if monotonic() - started_at > self._settings.stream_timeout_seconds:
                    raise LLMTimeoutError(
                        "Поток не завершился за отведенное время."
                    )

                model = chunk.model
                response_id = chunk.id

                if chunk.usage is not None:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens
                    total_tokens = chunk.usage.total_tokens

                if not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if choice.finish_reason is not None:
                    finish_reason = choice.finish_reason

                text = choice.delta.content
                if text:
                    yield LLMTextDelta(text=text)
        finally:
            stream.close()

        if model is None or response_id is None:
            raise LLMProviderError(
                "Провайдер завершил поток без метаданных."
            )

        yield LLMStreamCompleted(
            model=model,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            response_id=response_id,
        )

    @staticmethod
    def _convert_error(error: openai.APIError) -> LLMClientError:
        request_id = getattr(error, "request_id", None)

        if isinstance(error, openai.AuthenticationError):
            return LLMAccessError(
                "Провайдер отклонил API-ключ.",
                request_id=request_id,
            )
        if isinstance(error, openai.PermissionDeniedError):
            return LLMAccessError(
                "Недостаточно прав для обращения к модели.",
                request_id=request_id,
            )
        if isinstance(
                error,
                (
                        openai.BadRequestError,
                        openai.NotFoundError,
                        openai.UnprocessableEntityError,
                ),
        ):
            return LLMRequestError(
                "Провайдер отклонил модель или параметры запроса.",
                request_id=request_id,
            )
        if isinstance(error, openai.RateLimitError):
            return LLMRateLimitError(
                "Достигнуто ограничение частоты или квоты.",
                request_id=request_id,
            )
        if isinstance(error, openai.APITimeoutError):
            return LLMTimeoutError(
                "Провайдер не ответил за отведенное время."
            )
        if isinstance(error, openai.APIConnectionError):
            return LLMConnectionError(
                "Не удалось соединиться с провайдером."
            )
        if isinstance(error, openai.APIResponseValidationError):
            return LLMProviderError(
                "Провайдер вернул ответ неожиданного формата."
            )
        if isinstance(error, openai.APIStatusError):
            if error.status_code == 402:
                return LLMBalanceError(
                    "Недостаточно средств на балансе ProxyAPI.",
                    request_id=request_id,
                )
            return LLMProviderError(
                f"Провайдер вернул HTTP {error.status_code}.",
                request_id=request_id,
            )
        return LLMProviderError(
            "SDK сообщил о неизвестной ошибке провайдера."
        )

    def close(self) -> None:
        self._client.close()