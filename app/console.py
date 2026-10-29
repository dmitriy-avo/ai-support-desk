from uuid import uuid4

from app.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
    ChatStreamCompleted,
    ChatTextDelta,
)


def print_usage(result: ChatReply) -> None:
    if result.total_tokens is None:
        print("Провайдер не вернул статистику токенов.")
        return

    print(f"Входные токены: {result.prompt_tokens}")
    print(f"Выходные токены: {result.completion_tokens}")
    print(f"Всего токенов: {result.total_tokens}")


def print_metadata(result: ChatReply) -> None:
    print(f"Session ID ответа: {result.session_id}")
    print(f"Сообщений в истории: {result.history_messages}")
    print(
        "Сообщений истории в запросе: "
        f"{result.context_history_messages}"
    )
    print(
        "Сообщений истории отброшено: "
        f"{result.removed_history_messages}"
    )
    print(
        "Оценка входных токенов до запроса: "
        f"{result.estimated_prompt_tokens}"
    )
    print(
        "Summary сохранено: "
        f"{'да' if result.has_summary else 'нет'}"
    )
    print(
        "Summary обновлено на этом ходу: "
        f"{'да' if result.summary_updated else 'нет'}"
    )
    print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
    print(f"Время ответа: {result.elapsed_seconds:.2f} с")
    print_usage(result)
    print(f"ID ответа: {result.response_id}")


def run_console(
    service: ChatService,
    app_env: str,
    configured_model: str,
) -> None:
    print(f"Окружение: {app_env}")
    print(f"Настроенная модель: {configured_model}")

    session_id = uuid4().hex
    print(f"Session ID: {session_id}")
    print("Команды: /new — новый диалог, /exit — выход.")
    print("Во время ответа нажмите Ctrl+C, чтобы отменить генерацию.")

    while True:
        user_text = input("\nВы: ")
        command = user_text.strip()

        if command == "/exit":
            return

        if command == "/new":
            service.reset_session(session_id)
            session_id = uuid4().hex
            print(f"Новый Session ID: {session_id}")
            continue

        stream = service.stream_reply(session_id, user_text)
        result: ChatReply | None = None
        received_text = False

        print("Assistant: ", end="", flush=True)

        try:
            for event in stream:
                if isinstance(event, ChatTextDelta):
                    received_text = True
                    print(event.text, end="", flush=True)
                elif isinstance(event, ChatStreamCompleted):
                    result = event.result
        except KeyboardInterrupt:
            stream.close()
            print("\nГенерация отменена. Ответ не сохранен в истории.")
            continue
        except ChatServiceError as error:
            stream.close()
            print()
            if received_text:
                print(f"Поток прерван: {error}")
            else:
                print(f"Ошибка чата: {error}")
            continue

        print()

        if result is None:
            print("Ошибка чата: поток завершился без результата.")
            continue

        print_metadata(result)