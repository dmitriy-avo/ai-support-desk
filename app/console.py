# from app.support_service import SupportService, SupportServiceError, SupportSummary
#
#
# def print_usage(result: SupportSummary) -> None:
#     if result.total_tokens is None:
#         print("Провайдер не вернул статистику токенов.")
#         return
#
#     print(f"Входные токены: {result.prompt_tokens}")
#     print(f"Выходные токены: {result.completion_tokens}")
#     print(f"Всего токенов: {result.total_tokens}")
#
#
# def run_console(service: SupportService, app_env: str, configured_model: str) -> None:
#     print(f"Окружение: {app_env}")
#     print(f"Настроенная модель: {configured_model}")
#
#     user_text = input("Введите текст обращения: ")
#
#     try:
#         result = service.summarize(user_text)
#     except SupportServiceError as error:
#         print(f"Не удалось создать резюме: {error}")
#         return
#
#     print("\nРезюме обращения:")
#     print(result.text)
#     print("\nМетрики запроса:")
#     print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
#     print(f"Модель ответа: {result.model}")
#     print(f"Время ответа: {result.elapsed_seconds:.2f} с")
#     print_usage(result)
#     print(f"ID ответа: {result.response_id}")

from app.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
)


def print_usage(result: ChatReply) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")


def run_console(
        service: ChatService,
        app_env: str,
        configured_model: str,
) -> None:
    print(f"Окружение: {app_env}")
    print(f"Настроенная модель: {configured_model}")
    print("Для выхода введите /exit.")

    while True:
        user_text = input("\nВы: ")
        if user_text.strip() == "/exit":
            return

        try:
            result = service.reply(user_text)
        except ChatServiceError as error:
            print(f"Ошибка чата: {error}")
            continue

        print(f"Assistant: {result.text}")
        print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
        print(f"Время ответа: {result.elapsed_seconds:.2f} с")
        print_usage(result)
        print(f"ID ответа: {result.response_id}")
        print(
            f"Сообщений в истории: "
            f"{result.history_messages}"
        )