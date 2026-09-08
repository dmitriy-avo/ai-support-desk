import os

from dotenv import load_dotenv
from openai import OpenAI


BASE_URL = "https://api.mistral.ai/v1"


def main() -> None:
    load_dotenv()

    token = os.getenv("MISTRAL_API_KEY")
    if not token:
        raise SystemExit(
            "Не найдена переменная LLM_API_KEY. "
            "Проверьте файл .env в корне проекта."
        )

    client = OpenAI(
        base_url=BASE_URL,
        api_key=token,
    )

    print("Файл .env загружен, токен найден, клиент создан")


if __name__ == "__main__":
    main()