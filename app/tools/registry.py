from collections.abc import Callable
from dataclasses import dataclass
from re import fullmatch
from typing import Any

from pydantic import BaseModel



class ToolRegistryError(RuntimeError):
    pass


class DuplicateToolError(ToolRegistryError):
    pass


class ToolNotRegisteredError(ToolRegistryError):
    pass


@dataclass(frozen=True)
class RegisteredTool[ArgumentsT: BaseModel, ResultT: BaseModel]:
    name: str
    description: str
    arguments_model: type[ArgumentsT]
    handler: Callable[[ArgumentsT], ResultT]
    timeout_seconds: float = 2.0

    def __post_init__(self) -> None:
        if fullmatch(r"[a-z][a-z0-9_]{0,63}", self.name) is None:
            raise ValueError(
                "Имя инструмента должно начинаться "
                "с буквы и содержать только "
                "строчные буквы, цифры и _."
            )

        if not self.description.strip():
            raise ValueError("Описание инструмента не должно быть пустым.")

        if self.timeout_seconds <= 0:
            raise ValueError("Timeout инструмента должен быть больше нуля.")


    def to_api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "strict": True,
                "parameters": self.arguments_model.model_json_schema(),
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool[Any, Any]] = {}

    def register(self, tool: RegisteredTool[Any, Any]) -> None:
        if tool.name in self._tools:
            raise DuplicateToolError(f"Инструмент уже зарегистрирован: {tool.name}.")

        self._tools[tool.name] = tool

    def get(self, name: str) -> RegisteredTool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError as error:
            raise ToolNotRegisteredError(f"Инструмент не зарегистрирован: {name}.") from error

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def api_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_api_schema() for tool in self._tools.values()]