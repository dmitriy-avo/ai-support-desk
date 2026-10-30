from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter
from collections.abc import Generator

from app.chat_history import (
    ChatHistoryError,
    SlidingWindowHistory,
)
from app.llm.client import (
    LLMClient,
    LLMStreamCompleted,
    LLMTextDelta,
    Message,
)
from app.llm.errors import LLMClientError
from app.prompts.support_chat import SUPPORT_CHAT_PROMPT
from app.prompts.template import PromptTemplate
from app.conversation_summary_service import (
    ConversationSummaryError,
    ConversationSummaryService,
)




class ChatServiceError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatReply:
    text: str
    session_id: str
    history_messages: int
    context_history_messages: int
    removed_history_messages: int
    estimated_prompt_tokens: int
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    summary_updated: bool
    has_summary: bool

@dataclass(frozen=True)
class ChatTextDelta:
    text: str


@dataclass(frozen=True)
class ChatStreamCompleted:
    result: ChatReply


ChatStreamEvent = ChatTextDelta | ChatStreamCompleted


@dataclass
class ChatSession:
    messages: list[Message] = field(default_factory=list)
    summary: str | None = None
    lock: Lock = field(default_factory=Lock)


class ChatService:
    def __init__(
            self,
            llm_client: LLMClient,
            history_policy: SlidingWindowHistory,
            summary_service: ConversationSummaryService,
            prompt: PromptTemplate = SUPPORT_CHAT_PROMPT,
    ) -> None:
        self._llm_client = llm_client
        self._history_policy = history_policy
        self._summary_service = summary_service
        self._prompt = prompt
        self._chat_instruction = prompt.render()
        self._sessions: dict[str, ChatSession] = {}
        self._sessions_lock = Lock()

    def reply(self, session_id: str, user_text: str) -> ChatReply:
        result: ChatReply | None = None

        for event in self.stream_reply(session_id, user_text):
            if isinstance(event, ChatStreamCompleted):
                result = event.result

        if result is None:
            raise ChatServiceError(
                "Поток завершился без итогового результата."
            )

        return result

    def stream_reply(
            self,
            session_id: str,
            user_text: str,
    ) -> Generator[ChatStreamEvent, None, None]:
        session_key = self._normalize_session_id(session_id)

        text = user_text.strip()
        if not text:
            raise ChatServiceError("Сообщение не должно быть пустым.")

        session = self._get_or_create_session(session_key)

        with session.lock:
            developer_message = self._build_developer_message()
            user_message = self._build_user_message(text)
            summary_message = self._build_summary_message(session.summary)
            context_messages = (
                [summary_message] if summary_message is not None else []
            )

            try:
                prepared = self._history_policy.prepare(
                    developer_message=developer_message,
                    history=session.messages,
                    user_message=user_message,
                    context_messages=context_messages,
                )
            except ChatHistoryError as error:
                raise ChatServiceError(
                    f"Не удалось подготовить историю: {error}"
                ) from error

            started_at = perf_counter()
            new_summary = session.summary
            summary_updated = False

            if prepared.removed_history_messages > 0:
                try:
                    new_summary = self._summary_service.summarize(
                        previous_summary=session.summary,
                        messages=session.messages,
                    )
                except ConversationSummaryError as error:
                    raise ChatServiceError(
                        f"Не удалось сократить историю: {error}"
                    ) from error

                summary_updated = True
                summary_message = self._build_summary_message(new_summary)
                if summary_message is None:
                    raise ChatServiceError(
                        "Не удалось подготовить новое summary."
                    )

                try:
                    prepared = self._history_policy.prepare(
                        developer_message=developer_message,
                        history=[],
                        user_message=user_message,
                        context_messages=[summary_message],
                    )
                except ChatHistoryError as error:
                    raise ChatServiceError(
                        "Новое summary не помещается "
                        f"в бюджет: {error}"
                    ) from error

            reply_parts: list[str] = []
            llm_completed: LLMStreamCompleted | None = None

            try:
                for event in self._llm_client.stream(prepared.messages):
                    if isinstance(event, LLMTextDelta):
                        reply_parts.append(event.text)
                        yield ChatTextDelta(text=event.text)
                    elif isinstance(event, LLMStreamCompleted):
                        llm_completed = event
            except LLMClientError as error:
                raise ChatServiceError(
                    f"Не удалось получить ответ модели: {error}"
                ) from error

            elapsed_seconds = perf_counter() - started_at

            if llm_completed is None:
                raise ChatServiceError(
                    "Поток завершился без итоговых метаданных."
                )
            if llm_completed.finish_reason == "length":
                raise ChatServiceError(
                    "Ответ остановлен из-за ограничения длины."
                )
            if llm_completed.finish_reason != "stop":
                raise ChatServiceError(
                    "Модель не вернула готовый ответ. "
                    f"Причина завершения: {llm_completed.finish_reason}."
                )

            reply_text = self._validate_reply("".join(reply_parts))
            assistant_message: Message = {
                "role": "assistant",
                "content": reply_text,
            }

            if summary_updated:
                session.summary = new_summary
                session.messages = [user_message, assistant_message]
            else:
                session.messages.extend([user_message, assistant_message])

            history_messages = len(session.messages)

            result = ChatReply(
                text=reply_text,
                session_id=session_key,
                history_messages=history_messages,
                context_history_messages=prepared.context_history_messages,
                removed_history_messages=prepared.removed_history_messages,
                estimated_prompt_tokens=prepared.estimated_tokens,
                model=llm_completed.model,
                prompt_id=self._prompt.prompt_id,
                prompt_version=self._prompt.version,
                elapsed_seconds=elapsed_seconds,
                prompt_tokens=llm_completed.prompt_tokens,
                completion_tokens=llm_completed.completion_tokens,
                total_tokens=llm_completed.total_tokens,
                response_id=llm_completed.response_id,
                summary_updated=summary_updated,
                has_summary=session.summary is not None,
            )

        yield ChatStreamCompleted(result=result)

    def reset_session(self, session_id: str) -> None:
        session_key = self._normalize_session_id(session_id)

        with self._sessions_lock:
            session = self._sessions.get(session_key)

        if session is None:
            return

        with session.lock:
            session.messages.clear()
            session.summary = None

    def _get_or_create_session(
        self,
        session_id: str,
    ) -> ChatSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
            if session is None:
                session = ChatSession()
                self._sessions[session_id] = session
            return session

    @staticmethod
    def _normalize_session_id(session_id: str) -> str:
        session_key = session_id.strip()
        if not session_key:
            raise ChatServiceError("Session ID не должен быть пустым.")
        return session_key


    def _build_developer_message(self) -> Message:
        return {
            "role": "system",
            "content": self._chat_instruction,
        }

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
    def _build_summary_message(
        summary: str | None,
    ) -> Message | None:
        if summary is None:
            return None

        return {
            "role": "user",
            "content": (
                "<conversation_summary>\n"
                f"{summary}\n"
                "</conversation_summary>"
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