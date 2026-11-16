
from dataclasses import dataclass
from typing import Literal, Protocol

from app.llm.client import LLMResult, Message, ToolCall, ToolDefinition
from app.llm.errors import LLMClientError
from app.tools.execution import ToolAuditRecord, ToolExecutionError, ToolExecutor
from app.tools.registry import ToolRegistry


class ToolCallingClient(Protocol):
    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list[ToolDefinition],
        tool_choice: Literal["auto", "none"] = "auto",
    ) -> LLMResult:
        ...


class ToolLoopError(RuntimeError):
    pass


class ToolLoopProtocolError(ToolLoopError):
    pass


class ToolLoopLimitError(ToolLoopError):
    pass


class ToolLoopExecutionError(ToolLoopError):
    def __init__(self, message: str, audit: ToolAuditRecord) -> None:
        super().__init__(message)
        self.audit = audit


@dataclass(frozen=True)
class ToolLoopResult:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_ids: tuple[str, ...]
    model_steps: int
    tool_calls_count: int
    tool_names: tuple[str, ...]
    audit_records: tuple[ToolAuditRecord, ...]


class BoundedToolLoop:
    def __init__(
        self,
        client: ToolCallingClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        max_steps: int = 3,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps должен быть больше нуля.")

        self._client = client
        self._registry = registry
        self._executor = executor
        self._max_steps = max_steps

    def run(self, messages: list[Message]) -> ToolLoopResult:
        conversation = list(messages)
        llm_results: list[LLMResult] = []
        tool_names: list[str] = []
        audit_records: list[ToolAuditRecord] = []

        for step in range(1, self._max_steps + 1):
            try:
                result = self._client.generate_with_tools(
                    messages=conversation,
                    tools=self._registry.api_schemas(),
                    tool_choice="auto",
                )
            except LLMClientError as error:
                raise ToolLoopError(
                    f"Не удалось обратиться к модели: {error}"
                ) from error

            llm_results.append(result)

            if result.refusal is not None:
                raise ToolLoopProtocolError(
                    f"Модель отказалась отвечать: {result.refusal}"
                )

            if result.finish_reason == "stop":
                if result.tool_calls:
                    raise ToolLoopProtocolError(
                        "Текстовый ответ содержит tool calls."
                    )

                text = self._require_text(result)
                return self._build_result(
                    text=text,
                    llm_results=llm_results,
                    tool_names=tool_names,
                    audit_records=audit_records,
                )

            if result.finish_reason != "tool_calls":
                raise ToolLoopProtocolError(
                    f"Неожиданная причина завершения: {result.finish_reason}."
                )

            if len(result.tool_calls) != 1:
                raise ToolLoopProtocolError(
                    "Ожидался ровно один tool call на шаг."
                )

            if step == self._max_steps:
                raise ToolLoopLimitError(
                    "Модель запросила инструмент на последнем разрешённом шаге."
                )

            tool_call = result.tool_calls[0]
            conversation.append(
                self._build_tool_call_message(result, tool_call)
            )

            try:
                execution = self._executor.execute(tool_call)
            except ToolExecutionError as error:
                raise ToolLoopExecutionError(
                    str(error), error.audit
                ) from error

            tool_names.append(tool_call.name)
            audit_records.append(execution.audit)

            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": execution.content,
                }
            )

        raise ToolLoopLimitError(
            "Цикл завершился без финального ответа."
        )

    @staticmethod
    def _build_tool_call_message(
        result: LLMResult,
        tool_call: ToolCall,
    ) -> Message:
        return {
            "role": "assistant",
            "content": result.text,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    },
                }
            ],
        }

    @staticmethod
    def _require_text(result: LLMResult) -> str:
        if result.text is None:
            raise ToolLoopProtocolError("Модель не вернула текст.")

        text = result.text.strip()
        if not text:
            raise ToolLoopProtocolError("Модель вернула пустой текст.")

        return text

    @staticmethod
    def _build_result(
        text: str,
        llm_results: list[LLMResult],
        tool_names: list[str],
        audit_records: list[ToolAuditRecord],
    ) -> ToolLoopResult:

        def sum_usage(values: list[int | None]) -> int | None:
            if any(value is None for value in values):
                return None
            return sum(value for value in values if value is not None)

        return ToolLoopResult(
            text=text,
            model=llm_results[-1].model,
            prompt_tokens=sum_usage(
                [result.prompt_tokens for result in llm_results]
            ),
            completion_tokens=sum_usage(
                [result.completion_tokens for result in llm_results]
            ),
            total_tokens=sum_usage(
                [result.total_tokens for result in llm_results]
            ),
            response_ids=tuple(
                result.response_id for result in llm_results
            ),
            model_steps=len(llm_results),
            tool_calls_count=len(tool_names),
            tool_names=tuple(tool_names),
            audit_records=tuple(audit_records),
        )
