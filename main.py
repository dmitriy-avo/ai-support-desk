import os
import sys
from time import perf_counter

import openai
from dotenv import load_dotenv
from openai import OpenAI


def print_usage(usage) -> None:
    if usage is None:
        print("Провайдер не вернул статистику токенов")
        return

    print(f"Входные токены: {usage.prompt_tokens}")
    print(f"Выходные токены: {usage.completion_tokens}")
    print(f"Всего токенов: {usage.total_tokens}")


def build_messages(user_text: str, system_instructions: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": system_instructions,
        },
        {
            "role": "user",
            "content": f"Обращение:\n{user_text}",
        },
    ]

def validate_summary(summary: str | None) -> str:
    if summary is None:
        raise ValueError("Модель не вернула текст")

    cleaned = summary.strip()
    if not cleaned:
        raise ValueError("Модель вернула пустой ответ")
    if len(cleaned) > 300:
        raise ValueError("Резюме получилось слишком длинным")

    return cleaned


def summarize_request(client: OpenAI,
                      model: str,
                      user_text: str,
                      system_instructions: str,
                      temperature: float,
                      max_completion_tokens: int,) -> None:
    """Кратко пересказывает обращение и печатает метрики запроса."""
    text = user_text.strip()
    if not text:
        print("Ошибка: обращение не должно быть пустым.")
        return

    started_at = perf_counter()

    try:
        response = client.chat.completions.create(
            model=model,
            messages=build_messages(user_text, system_instructions),
            temperature=temperature,
            max_tokens=max_completion_tokens,
        )
    except openai.AuthenticationError:
        print("Ошибка авторизации: проверьте LLM_API_KEY.")
        return
    except openai.PermissionDeniedError:
        print("Нет доступа к модели: проверьте разрешения API-ключа.")
        return
    except openai.RateLimitError:
        print("Достигнут лимит запросов провайдера. Повторите запрос позднее.")
        return
    except openai.APITimeoutError:
        print("Провайдер не успел ответить за отведенное время.")
        return
    except openai.APIConnectionError:
        print("Не удалось соединиться с провайдером. Проверьте сеть.")
        return
    except openai.APIStatusError as error:
        if error.status_code == 402:
            print("Недостаточно средств на балансе провайдера.")
        else:
            print(f"API вернул ошибку со статусом {error.status_code}.")
        if error.request_id:
            print(f"Request ID: {error.request_id}")
        return

    elapsed_seconds = perf_counter() - started_at
    choice = response.choices[0]

    if choice.finish_reason == "length":
        print(
            "Ответ модели остановлен из-за ограничения длины. "
            "Увеличьте MAX_OUTPUT_TOKENS или сократите задачу."
        )
        return

    if choice.finish_reason != "stop":
        print(
            "Модель не вернула готовое резюме. "
            f"Причина завершения: {choice.finish_reason}."
        )
        return

    try:
        answer = validate_summary(choice.message.content)
    except ValueError as error:
        print(f"Некорректный ответ модели: {error}")
        return

    print("\nРезультат:")
    print(answer)

    print("\nМетрики:")
    print(f"Модель: {response.model}")
    print(f"Завершение: {choice.finish_reason}")
    print(f"Время: {elapsed_seconds:.2f} с")

    print_usage(response.usage)

    print(f"ID ответа: {response.id}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")
    token = os.getenv("MISTRAL_API_KEY", "ollama")
    system_instructions = os.getenv("SYSTEM_INSTRUCTION", "")
    temperature = float(os.getenv("TEMPERATURE"))
    max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS"))

    client = OpenAI(
        base_url=base_url,
        api_key=token,
    )

    user_text = input("Введите текст обращения: ")
    summarize_request(client, model, user_text, system_instructions, temperature, max_output_tokens)


if __name__ == "__main__":
    main()