import os
import sys

from dotenv import load_dotenv
from openai import OpenAI


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv()

    model = os.getenv("MODEL")
    base_url = os.getenv("BASE_URL")
    token = os.getenv("MISTRAL_API_KEY", "ollama")

    client = OpenAI(
        base_url=base_url,
        api_key=token,
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Кратко перескажи: клиент дважды оплатил один заказ.",
            }
        ],
    )

    choice = response.choices[0]
    answer = choice.message.content

    print("Результат:")
    print(answer or "Модель не вернула текст")

    print("\nДиагностика:")
    print(f"ID ответа: {response.id}")
    print(f"Модель: {response.model}")
    print(f"Причина завершения: {choice.finish_reason}")

    if response.usage is not None:
        print(f"Входные токены: {response.usage.prompt_tokens}")
        print(f"Выходные токены: {response.usage.completion_tokens}")
        print(f"Всего токенов: {response.usage.total_tokens}")
    else:
        print("Провайдер не вернул статистику токенов")


if __name__ == "__main__":
    main()