from dataclasses import dataclass
from time import perf_counter

from pydantic import ValidationError

from app.llm.client import LLMClient, LLMResult, Message
from app.llm.errors import LLMClientError
from app.models.support_request import SupportRequestAnalysis
from app.models.support_request_validation import (
    SupportRequestSemanticError,
    validate_support_request,
)
from app.prompts.support_request_analysis import (
    SUPPORT_REQUEST_ANALYSIS_PROMPT,
)
from app.schemas.support_request import (
    SUPPORT_REQUEST_SCHEMA,
    SUPPORT_REQUEST_SCHEMA_NAME,
)


DEVELOPER_INSTRUCTION = SUPPORT_REQUEST_ANALYSIS_PROMPT.render()
MAX_REPAIR_ATTEMPTS = 1

class SupportRequestAnalyzerError(RuntimeError):
    pass


class SupportRequestRefusalError(SupportRequestAnalyzerError):
    pass


class IncompleteSupportRequestError(SupportRequestAnalyzerError):
    pass


class InvalidSupportRequestError(SupportRequestAnalyzerError):
    pass


@dataclass(frozen=True)
class SupportRequestResult:
    analysis: SupportRequestAnalysis
    model: str
    prompt_id: str
    prompt_version: str
    elapsed_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    response_id: str
    repair_attempts: int

class SupportRequestAnalyzer:
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm_client = llm_client

    def analyze(self, user_text: str) -> SupportRequestResult:
        text = user_text.strip()
        if not text:
            raise SupportRequestAnalyzerError(
                "Обращение не должно быть пустым."
            )

        started_at = perf_counter()

        try:
            analysis, llm_result, repair_attempts = self._request_analysis(text)
        except LLMClientError as error:
            raise SupportRequestAnalyzerError(
                f"Не удалось получить анализ обращения: {error}"
            ) from error

        return SupportRequestResult(
            analysis=analysis,
            model=llm_result.model,
            prompt_id=SUPPORT_REQUEST_ANALYSIS_PROMPT.prompt_id,
            prompt_version=SUPPORT_REQUEST_ANALYSIS_PROMPT.version,
            elapsed_seconds=perf_counter() - started_at,
            prompt_tokens=llm_result.prompt_tokens,
            completion_tokens=llm_result.completion_tokens,
            total_tokens=llm_result.total_tokens,
            response_id=llm_result.response_id,
            repair_attempts=repair_attempts,
        )

    @staticmethod
    def _build_messages(user_text: str) -> list[Message]:
        return [
            {
                "role": "system",
                "content": DEVELOPER_INSTRUCTION,
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

    @classmethod
    def _build_repair_messages(cls, user_text: str, issues: tuple[str, ...]) -> list[Message]:
        messages = cls._build_messages(user_text)
        issue_text = "\n".join(f"- {issue}" for issue in issues)

        messages[0] = {
            "role": "system",
            "content": (
                f"{DEVELOPER_INSTRUCTION}\n\n"
                "Предыдущий результат не прошел проверку.\n"
                "Исправь перечисленные противоречия:\n"
                f"{issue_text}"
            ),
        }
        return messages

    @staticmethod
    def _parse_result(result: LLMResult) -> SupportRequestAnalysis:
        if result.refusal is not None:
            raise SupportRequestRefusalError(result.refusal)

        if result.finish_reason == "length":
            raise IncompleteSupportRequestError(
                "Модели не хватило выходных токенов."
            )

        if result.finish_reason != "stop":
            raise InvalidSupportRequestError(
                "Неожиданная причина завершения: "
                f"{result.finish_reason}."
            )

        if result.text is None:
            raise InvalidSupportRequestError(
                "Модель не вернула данные."
            )

        try:
            return SupportRequestAnalysis.model_validate_json(result.text)
        except ValidationError as error:
            raise InvalidSupportRequestError(
                f"Ответ модели не соответствует контракту: {error}"
            ) from error

    def _request_analysis(
        self,
        user_text: str,
    ) -> tuple[SupportRequestAnalysis, LLMResult, int]:
        messages = self._build_messages(user_text)

        for repair_attempts in range(
            MAX_REPAIR_ATTEMPTS + 1
        ):
            result = self._llm_client.generate_structured(
                messages=messages,
                schema_name=SUPPORT_REQUEST_SCHEMA_NAME,
                schema=SUPPORT_REQUEST_SCHEMA,
            )
            analysis = self._parse_result(result)

            try:
                validate_support_request(analysis, user_text)
            except SupportRequestSemanticError as error:
                if repair_attempts == MAX_REPAIR_ATTEMPTS:
                    issue_text = " ".join(error.issues)
                    raise InvalidSupportRequestError(
                        "Результат не прошел смысловую "
                        f"проверку: {issue_text}"
                    ) from error

                messages = self._build_repair_messages(
                    user_text,
                    error.issues,
                )
                continue

            return analysis, result, repair_attempts

        raise RuntimeError("Недостижимое состояние.")