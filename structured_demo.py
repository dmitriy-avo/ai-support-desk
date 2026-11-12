import json

from pydantic import ValidationError

from app.config import Settings
from app.llm.client import LLMClient, LLMResult, Message
from app.llm.errors import LLMClientError
from app.schemas.support_request import (
    SUPPORT_REQUEST_SCHEMA,
    SUPPORT_REQUEST_SCHEMA_NAME,
)


DEVELOPER_INSTRUCTION = """
Проанализируй обращение клиента службы поддержки.
Используй только сведения из обращения.
Не придумывай номер заказа, товар и другие отсутствующие факты.
Определи категорию, срочность и недостающие сведения.
""".strip()


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


def print_result(result: LLMResult) -> None:
    if result.refusal is not None:
        print(f"Модель отказалась отвечать: {result.refusal}")
        return

    if result.finish_reason == "length":
        print("Ответ остановлен из-за ограничения длины.")
        return

    if result.finish_reason != "stop":
        print(
            "Генерация завершилась с неожиданной причиной: "
            f"{result.finish_reason}"
        )
        return

    if result.text is None:
        print("Модель не вернула структурированный результат.")
        return

    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError:
        print("Провайдер вернул строку с некорректным JSON.")
        return

    print("\nСтруктурированный результат:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nМодель: {result.model}")
    print(f"Причина завершения: {result.finish_reason}")
    print(f"Всего токенов: {result.total_tokens}")
    print(f"ID ответа: {result.response_id}")


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
        result = llm_client.generate_structured(
            messages=build_messages(user_text),
            schema_name=SUPPORT_REQUEST_SCHEMA_NAME,
            schema=SUPPORT_REQUEST_SCHEMA,
        )
    except LLMClientError as error:
        print(f"Не удалось проанализировать обращение: {error}")
        return
    finally:
        llm_client.close()

    print_result(result)


if __name__ == "__main__":
    main()