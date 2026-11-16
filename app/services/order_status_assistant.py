from dataclasses import dataclass

from pydantic import ValidationError

from app.llm.client import (
    LLMClient,
    LLMResult,
    Message,
    ToolCall,
)
from app.llm.errors import LLMClientError
from app.prompts.order_status_assistant import (
    ORDER_STATUS_ASSISTANT_PROMPT,
)
from app.tools.order_status import (
    GET_ORDER_STATUS_TOOL_NAME,
    GetOrderStatusArguments,
    ORDER_STATUS_TOOL,
    get_order_status,
)

class OrderStatusAssistantError(RuntimeError):
    pass


class UnknownToolError(OrderStatusAssistantError):
    pass


class InvalidToolArgumentsError(OrderStatusAssistantError):
    pass


class UnexpectedToolFlowError(OrderStatusAssistantError):
    pass


@dataclass(frozen=True)
class OrderStatusAnswer:
    text: str
    tool_used: bool
    order_id: str | None
    model: str
    prompt_id: str
    prompt_version: str
    total_tokens: int | None
    response_ids: tuple[str, ...]


class OrderStatusAssistant:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client
        self._instruction = (
            ORDER_STATUS_ASSISTANT_PROMPT.render()
        )

    def _build_messages(
        self,
        user_text: str,
    ) -> list[Message]:
        return [
            {
                "role": "system",
                "content": self._instruction,
            },
            {
                "role": "user",
                "content": (
                    "<customer_request>\n"
                    f"{user_text}\n"
                    "</customer_request>"
                ),
            },
        ]

    def answer(self, user_text: str) -> OrderStatusAnswer:
        text = user_text.strip()
        if not text:
            raise OrderStatusAssistantError(
                "Запрос не должен быть пустым."
            )

        messages = self._build_messages(text)

        try:
            first_result = self._llm_client.generate_with_tools(
                messages=messages,
                tools=[ORDER_STATUS_TOOL.to_api_schema()],
                tool_choice="auto",
            )
        except LLMClientError as error:
            raise OrderStatusAssistantError(
                f"Не удалось обработать запрос: {error}"
            ) from error

        if first_result.refusal is not None:
            raise OrderStatusAssistantError(
                f"Модель отказалась отвечать: "
                f"{first_result.refusal}"
            )

        if first_result.finish_reason == "stop":
            if first_result.tool_calls:
                raise UnexpectedToolFlowError(
                    "Текстовый ответ неожиданно содержит tool calls."
                )

            reply = self._require_text(first_result)
            return self._build_answer(
                text=reply,
                tool_used=False,
                order_id=None,
                results=(first_result,),
            )

        if first_result.finish_reason != "tool_calls":
            raise UnexpectedToolFlowError(
                "Неожиданная причина завершения: "
                f"{first_result.finish_reason}."
            )

        if len(first_result.tool_calls) != 1:
            raise UnexpectedToolFlowError(
                "Ожидался ровно один tool call."
            )

        tool_call = first_result.tool_calls[0]

        if tool_call.name != GET_ORDER_STATUS_TOOL_NAME:
            raise UnknownToolError(
                f"Неизвестный инструмент: {tool_call.name}."
            )

        try:
            arguments = (
                GetOrderStatusArguments.model_validate_json(
                    tool_call.arguments
                )
            )
        except ValidationError as error:
            raise InvalidToolArgumentsError(
                "Модель вернула некорректные аргументы "
                "для get_order_status."
            ) from error

        tool_result = get_order_status(arguments)

        messages.extend(
            [
                self._build_tool_call_message(
                    first_result,
                    tool_call,
                ),
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_result.model_dump_json(),
                },
            ]
        )

        try:
            final_result = self._llm_client.generate_with_tools(
                messages=messages,
                tools=[ORDER_STATUS_TOOL.to_api_schema()],
                tool_choice="none",
            )
        except LLMClientError as error:
            raise OrderStatusAssistantError(
                f"Не удалось сформировать ответ: {error}"
            ) from error

        if final_result.refusal is not None:
            raise OrderStatusAssistantError(
                f"Модель отказалась отвечать: "
                f"{final_result.refusal}"
            )

        if final_result.finish_reason != "stop":
            raise UnexpectedToolFlowError(
                "Модель не вернула финальный текст. "
                f"Причина: {final_result.finish_reason}."
            )

        if final_result.tool_calls:
            raise UnexpectedToolFlowError(
                "Финальный ответ содержит новый tool call."
            )

        reply = self._require_text(final_result)
        return self._build_answer(
            text=reply,
            tool_used=True,
            order_id=arguments.order_id,
            results=(first_result, final_result),
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
            raise UnexpectedToolFlowError(
                "Модель не вернула текст."
            )

        text = result.text.strip()
        if not text:
            raise UnexpectedToolFlowError(
                "Модель вернула пустой текст."
            )
        return text

    @staticmethod
    def _build_answer(
            text: str,
            tool_used: bool,
            order_id: str | None,
            results: tuple[LLMResult, ...],
    ) -> OrderStatusAnswer:
        token_values = [result.total_tokens for result in results]
        total_tokens = (
            None
            if any(value is None for value in token_values)
            else sum(
                value
                for value in token_values
                if value is not None
            )
        )

        return OrderStatusAnswer(
            text=text,
            tool_used=tool_used,
            order_id=order_id,
            model=results[-1].model,
            prompt_id=ORDER_STATUS_ASSISTANT_PROMPT.prompt_id,
            prompt_version=(
                ORDER_STATUS_ASSISTANT_PROMPT.version
            ),
            total_tokens=total_tokens,
            response_ids=tuple(
                result.response_id for result in results
            ),
        )
