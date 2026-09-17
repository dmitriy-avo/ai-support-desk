from dataclasses import dataclass
from time import perf_counter

from app.llm.client import LLMClient, Message
from app.llm.errors import LLMClientError
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT

CHAT_INSTRUCTION = SUPPORT_CHAT_PROMPT.render()


class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatReply:
    text: str
    history_messages: int
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str


class ChatService:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._history: list[Message] = []

    def reply(self, user_text: str) -> ChatReply:
        text = user_text.strip()
        if not text:
            raise ChatServiceError("Сообщение не должно быть пустым.")

        user_message = self._build_user_message(text)
        messages = self._build_messages(user_message)

        started_at = perf_counter()
        try:
            llm_result = self._llm_client.generate(messages)
        except LLMClientError as error:
            raise ChatServiceError(
                f"Не удалось получить ответ модели: {error}"
            ) from error
        elapsed_seconds = perf_counter() - started_at

        if llm_result.finish_reason == "length":
            raise ChatServiceError("Ответ остановлен из-за ограничения длины.")
        if llm_result.finish_reason != "stop":
            raise ChatServiceError(
                "Модель не вернула готовый ответ. "
                f"Причина завершения: {llm_result.finish_reason}."
            )

        reply_text = self._validate_reply(llm_result.text)

        assistant_message: Message = {
            "role": "assistant",
            "content": reply_text,
        }
        self._history.extend([user_message, assistant_message])

        return ChatReply(
            text=reply_text,
            history_messages=len(self._history),
            model=llm_result.model,
            prompt_id=SUPPORT_CHAT_PROMPT.prompt_id,
            prompt_version=SUPPORT_CHAT_PROMPT.version,
            elapsed_seconds=elapsed_seconds,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
        )

    def _build_messages(
            self,
            user_message: Message,
    ) -> list[Message]:
        return [
            {
                "role": "system",
                "content": CHAT_INSTRUCTION,
            },
            *self._history,
            user_message,
        ]

    @staticmethod
    def _build_user_message(user_text: str) -> Message:
        return {
            "role": "user",
            "content": (
                "<customer_message>\n"
                f"{user_text}\n"
                "</customer_message>"
            ),
        }

    @staticmethod
    def _validate_reply(reply: str | None) -> str:
        if reply is None:
            raise ChatServiceError("Модель не вернула текст ответа.")

        cleaned_reply = reply.strip()
        if not cleaned_reply:
            raise ChatServiceError("Модель вернула пустой ответ.")

        return cleaned_reply