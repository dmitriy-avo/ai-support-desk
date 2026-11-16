from uuid import uuid4

from app.services.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
    ChatStreamCompleted,
    ChatTextDelta,
)

from app.services.support_request_analyzer import (
    SupportRequestAnalyzer,
    SupportRequestAnalyzerError,
)
from app.services.support_request_review import (
    SupportRequestDecision,
    decide_review,
)

from app.services.order_status_assistant import (
    OrderStatusAnswer,
    OrderStatusAssistant,
    OrderStatusAssistantError,
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


def print_analysis(
    decision: SupportRequestDecision,
) -> None:
    result = decision.result

    print("\nАнализ обращения:")
    print(result.analysis.model_dump_json(indent=2))
    print(
        "Нужна проверка человеком: "
        f"{'да' if decision.requires_human_review else 'нет'}"
    )

    if decision.review_reasons:
        reasons = ", ".join(
            reason.value for reason in decision.review_reasons
        )
        print(f"Причины проверки: {reasons}")

    print(f"Промпт: {result.prompt_id}@{result.prompt_version}")
    print(f"Исправляющих запросов: {result.repair_attempts}")
    print(f"Время анализа: {result.elapsed_seconds:.2f} с")
    print(f"Всего токенов: {result.total_tokens}")
    print(f"ID ответа: {result.response_id}")


def print_order_status_answer(result: OrderStatusAnswer) -> None:
    print(f"\nAssistant: {result.text}")
    print(
        "Инструмент использован: "
        f"{'да' if result.tool_used else 'нет'}"
    )

    if result.order_id is not None:
        print(f"Номер заказа: {result.order_id}")

    print(
        f"Промпт: {result.prompt_id}@{result.prompt_version}"
    )
    print(f"Всего токенов: {result.total_tokens}")
    print(
        "ID ответов: "
        f"{', '.join(result.response_ids)}"
    )



def run_console(
    service: ChatService,
    analyzer: SupportRequestAnalyzer,
    order_status_assistant: OrderStatusAssistant,
    app_env: str,
    configured_model: str,
) -> None:
    print(f"Окружение: {app_env}")
    print(f"Настроенная модель: {configured_model}")

    session_id = uuid4().hex
    print(f"Session ID: {session_id}")
    print(
        "Команды: /new — новый диалог, "
        "/analyze — анализ обращения, "
        "/order — статус заказа, /exit — выход."
    )

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

        if command == "/analyze":
            request_text = input("Обращение: ")

            try:
                analysis_result = analyzer.analyze(request_text)
            except SupportRequestAnalyzerError as error:
                print(f"Ошибка анализа: {error}")
                continue

            decision = decide_review(analysis_result)
            print_analysis(decision)
            continue

        if command == "/order":
            request_text = input("Вопрос о заказе: ")

            try:
                order_result = order_status_assistant.answer(
                    request_text
                )
            except OrderStatusAssistantError as error:
                print(f"Ошибка проверки заказа: {error}")
                continue

            print_order_status_answer(order_result)
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