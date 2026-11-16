from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
import json
from time import perf_counter

from pydantic import BaseModel, ValidationError

from app.llm.client import ToolCall
from app.tools.registry import ToolNotRegisteredError, ToolRegistry

class ToolBusinessError(RuntimeError):
    def __init__(self, message: str, code: str = "business_error") -> None:
        super().__init__(message)
        self.code = code

class ToolExecutionError(RuntimeError):
    def __init__(self, message: str, audit: "ToolAuditRecord") -> None:
        super().__init__(message)
        self.audit = audit


class ToolNotAllowedError(ToolExecutionError):
    pass


class ToolArgumentsError(ToolExecutionError):
    pass


class ToolTimeoutError(ToolExecutionError):
    pass


class ToolTechnicalError(ToolExecutionError):
    pass


class ToolExecutionStatus(StrEnum):
    SUCCESS = "success"
    BUSINESS_ERROR = "business_error"
    REJECTED = "rejected"
    TIMEOUT = "timeout"
    TECHNICAL_ERROR = "technical_error"


@dataclass(frozen=True)
class ToolAuditRecord:
    call_id: str
    tool_name: str
    started_at: str
    duration_seconds: float
    status: ToolExecutionStatus
    error_code: str | None

@dataclass(frozen=True)
class ToolExecutionResult:
    content: str
    audit: ToolAuditRecord

class ToolExecutor:
    def __init__(self, registry: ToolRegistry, max_workers: int = 4) -> None:
        if max_workers <= 0:
            raise ValueError("Количество workers должно быть больше нуля.")

        self._registry = registry
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool-executor")


    def execute(self, tool_call: ToolCall) -> ToolExecutionResult:
        started_at = datetime.now(UTC).isoformat()
        started = perf_counter()

        try:
            tool = self._registry.get(tool_call.name)
        except ToolNotRegisteredError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.REJECTED,
                error_code="unknown_tool",
            )
            raise ToolNotAllowedError("Модель запросила неизвестный инструмент.", audit) from error

        try:
            arguments = tool.arguments_model.model_validate_json(tool_call.arguments)
        except ValidationError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.REJECTED,
                error_code="invalid_arguments",
            )
            raise ToolArgumentsError("Аргументы инструмента не прошли проверку.", audit) from error

        future = self._pool.submit(tool.handler, arguments)

        try:
            result = future.result(timeout=tool.timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.TIMEOUT,
                error_code="timeout",
            )
            raise ToolTimeoutError("Инструмент не завершился вовремя.", audit) from error
        except ToolBusinessError as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.BUSINESS_ERROR,
                error_code=error.code,
            )
            return ToolExecutionResult(
                content=json.dumps(
                    {
                        "ok": False,
                        "error": {
                            "code": error.code,
                            "message": str(error),
                        },
                    },
                    ensure_ascii=False,
                ),
                audit=audit,
            )
        except Exception as error:
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.TECHNICAL_ERROR,
                error_code="technical_error",
            )
            raise ToolTechnicalError("Инструмент завершился технической ошибкой.", audit) from error

        if not isinstance(result, BaseModel):
            audit = self._build_audit(
                tool_call=tool_call,
                started_at=started_at,
                started=started,
                status=ToolExecutionStatus.TECHNICAL_ERROR,
                error_code="invalid_result_type",
            )
            raise ToolTechnicalError("Инструмент вернул результат неверного типа.", audit)

        audit = self._build_audit(
            tool_call=tool_call,
            started_at=started_at,
            started=started,
            status=ToolExecutionStatus.SUCCESS,
            error_code=None,
        )
        return ToolExecutionResult(
            content=json.dumps(
                {
                    "ok": True,
                    "data": result.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ),
            audit=audit,
        )

    @staticmethod
    def _build_audit(
            tool_call: ToolCall,
            started_at: str,
            started: float,
            status: ToolExecutionStatus,
            error_code: str | None,
    ) -> ToolAuditRecord:
        return ToolAuditRecord(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            started_at=started_at,
            duration_seconds=perf_counter() - started,
            status=status,
            error_code=error_code,
        )

    def close(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=True)
        