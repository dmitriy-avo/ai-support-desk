import json

from pydantic import ValidationError
from app.models.support_request import SupportRequestAnalysis
from app.config import Settings
from app.llm.client import LLMClient, LLMResult, Message
from app.llm.errors import LLMClientError
from app.schemas.support_request import (
    SUPPORT_REQUEST_SCHEMA,
    SUPPORT_REQUEST_SCHEMA_NAME,
)

from app.models.support_request_validation import (
    SupportRequestSemanticError,
    validate_support_request,
)

DEVELOPER_INSTRUCTION = """
Проанализируй обращение клиента службы поддержки.
Используй только сведения из обращения.
Не придумывай номер заказа, товар и другие отсутствующие факты.
Определи категорию, срочность и недостающие сведения.
""".strip()

MAX_REPAIR_ATTEMPTS = 1

class StructuredResultError(RuntimeError):
    pass


class ModelRefusalError(StructuredResultError):
    pass


class IncompleteStructuredResultError(StructuredResultError):
    pass


class StructuredResultValidationError(StructuredResultError):
    pass

def build_messages(user_text: str) -> list[Message]:
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

def build_repair_messages(
    user_text: str,
    issues: tuple[str, ...],
) -> list[Message]:
    messages = build_messages(user_text)
    issue_text = "\n".join(f"- {issue}" for issue in issues)

    messages[0] = {
        "role": "developer",
        "content": (
            f"{DEVELOPER_INSTRUCTION}\n\n"
            "Предыдущий результат не прошел проверку.\n"
            "Исправь перечисленные противоречия:\n"
            f"{issue_text}"
        ),
    }
    return messages

def parse_result(
    result: LLMResult,
) -> SupportRequestAnalysis:
    if result.refusal is not None:
        raise ModelRefusalError(result.refusal)

    if result.finish_reason == "length":
        raise IncompleteStructuredResultError(
            "Модели не хватило разрешенного количества токенов."
        )

    if result.finish_reason != "stop":
        raise StructuredResultError(
            "Неожиданная причина завершения генерации: "
            f"{result.finish_reason}."
        )

    if result.text is None:
        raise StructuredResultValidationError(
            "Модель не вернула структурированный результат."
        )

    try:
        return SupportRequestAnalysis.model_validate_json(result.text)
    except ValidationError as error:
        raise StructuredResultValidationError(
            "Ответ модели не соответствует контракту."
        ) from error

def request_analysis(
    llm_client: LLMClient,
    user_text: str,
) -> tuple[SupportRequestAnalysis, LLMResult, int]:
    messages = build_messages(user_text)

    for repair_attempts in range(MAX_REPAIR_ATTEMPTS + 1):
        result = llm_client.generate_structured(
            messages=messages,
            schema_name=SUPPORT_REQUEST_SCHEMA_NAME,
            schema=SUPPORT_REQUEST_SCHEMA,
        )
        analysis = parse_result(result)

        try:
            validate_support_request(analysis, user_text)
        except SupportRequestSemanticError as error:
            if repair_attempts == MAX_REPAIR_ATTEMPTS:
                raise

            messages = build_repair_messages(
                user_text,
                error.issues,
            )
            continue

        return analysis, result, repair_attempts

    raise RuntimeError("Недостижимое состояние repair flow.")

def print_result(
        analysis: SupportRequestAnalysis,
        result: LLMResult,
        repair_attempts: int,
) -> None:
    print("\nПроверенный результат:")
    print(
        json.dumps(
            analysis.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nМодель: {result.model}")
    print(f"Причина завершения: {result.finish_reason}")
    print(f"Всего токенов: {result.total_tokens}")
    print(f"ID ответа: {result.response_id}")
    print(f"Исправляющих запросов: {repair_attempts}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    user_text = input("Введите обращение клиента: ").strip()
    if not user_text:
        print("Обращение не должно быть пустым.")
        return

    llm_client = LLMClient(settings)
    try:
        analysis, result, repair_attempts = request_analysis(
            llm_client,
            user_text,
        )
    except LLMClientError as error:
        print(f"Не удалось обратиться к модели: {error}")
        return
    except ModelRefusalError as error:
        print(f"Модель отказалась анализировать обращение: {error}")
        return
    except IncompleteStructuredResultError as error:
        print(f"Получен незавершенный ответ: {error}")
        return
    except StructuredResultValidationError as error:
        print(f"Ответ не прошел проверку: {error}")
        return
    except StructuredResultError as error:
        print(f"Не удалось обработать результат: {error}")
        return
    except SupportRequestSemanticError as error:
        print("Результат не прошел смысловую проверку:")
        for issue in error.issues:
            print(f"- {issue}")
        return
    finally:
        llm_client.close()

    print_result(analysis, result, repair_attempts)


if __name__ == "__main__":
    main()