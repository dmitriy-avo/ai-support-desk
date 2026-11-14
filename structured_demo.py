import json

from pydantic import ValidationError

from app.config import Settings
from app.llm.client import LLMClient
from app.services.support_request_analyzer import (
    SupportRequestAnalyzer,
    SupportRequestAnalyzerError,
    SupportRequestResult,
)


def print_result(result: SupportRequestResult) -> None:
    print("\nПроверенный результат:")
    print(
        json.dumps(
            result.analysis.model_dump(mode="json"),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nМодель: {result.model}")
    print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
    print(f"Всего токенов: {result.total_tokens}")
    print(f"ID ответа: {result.response_id}")
    print(f"Исправляющих запросов: {result.repair_attempts}")


def main() -> None:
    try:
        settings = Settings()
    except ValidationError as error:
        print("Ошибка конфигурации:")
        for issue in error.errors():
            field = ".".join(str(part) for part in issue["loc"])
            print(f"- {field}: {issue['msg']}")
        return

    llm_client = LLMClient(settings)
    analyzer = SupportRequestAnalyzer(llm_client)
    user_text = input("Введите обращение клиента: ")

    try:
        result = analyzer.analyze(user_text)
    except SupportRequestAnalyzerError as error:
        print(f"Не удалось выполнить анализ: {error}")
        return
    finally:
        llm_client.close()

    print_result(result)


if __name__ == "__main__":
    main()