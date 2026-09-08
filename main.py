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
                "content": (
                    "Сформулируй в одном предложении краткое содержание обращения: "
                    "Покупатель получил поврежденную упаковку и хочет заменить товар."
                ),
            }
        ],
    )

    answer = response.choices[0].message.content
    print(answer)


if __name__ == "__main__":
    main()