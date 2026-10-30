from uuid import uuid4

from app.chat_service import (
    ChatReply,
    ChatService,
    ChatServiceError,
)
from evals.chat_cases import CHAT_EVAL_CASES
from evals.chat_models import (
    ChatEvalCase,
    TurnEvalResult,
    TurnExpectation,
)


def normalize_exact_answer(text: str) -> str:
    stripped = text.casefold().strip(" \t\n.,!?;:«»\"'")
    return " ".join(stripped.split())


def evaluate_reply(
    reply: ChatReply,
    expectation: TurnExpectation,
) -> tuple[str, ...]:
    failures: list[str] = []
    reply_text = reply.text
    lowered_text = reply_text.casefold()

    for fragment in expectation.required_fragments:
        if fragment.casefold() not in lowered_text:
            failures.append(
                f"В ответе нет обязательного фрагмента: {fragment!r}."
            )

    normalized_reply = normalize_exact_answer(reply_text)
    for forbidden in expectation.forbidden_exact_answers:
        if normalized_reply == normalize_exact_answer(forbidden):
            failures.append(
                f"Получен запрещенный точный ответ: {forbidden!r}."
            )

    if expectation.question_count is not None:
        actual_count = reply_text.count("?")
        if actual_count != expectation.question_count:
            failures.append(
                "Ожидалось вопросительных знаков: "
                f"{expectation.question_count}, получено: {actual_count}."
            )

    if len(reply_text) > expectation.max_characters:
        failures.append(
            f"Ответ длиннее {expectation.max_characters} символов."
        )

    return tuple(failures)


def run_case(
    chat_service: ChatService,
    case: ChatEvalCase,
) -> list[TurnEvalResult]:
    session_id = f"eval-{case.case_id}-{uuid4().hex}"
    results: list[TurnEvalResult] = []

    print(f"\nСценарий: {case.case_id}")
    print(case.description)
    print(f"Ручная проверка: {case.manual_review}")

    for turn_number, turn in enumerate(case.turns, start=1):
        reply: ChatReply | None = None
        error_text: str | None = None

        try:
            reply = chat_service.reply(session_id, turn.user_text)
        except ChatServiceError as error:
            error_text = str(error)

        expected_error = turn.expectation.expected_error_fragment

        if error_text is not None:
            if expected_error is None:
                failures = (f"Неожиданная ошибка: {error_text}",)
            elif expected_error.casefold() not in error_text.casefold():
                failures = (
                    "Получена другая ошибка. "
                    f"Ожидался фрагмент: {expected_error!r}.",
                )
            else:
                failures = ()
        elif reply is None:
            failures = ("Нет ни ответа, ни прикладной ошибки.",)
        elif expected_error is not None:
            failures = (
                f"Ожидалась ошибка с фрагментом: {expected_error!r}.",
            )
        else:
            failures = evaluate_reply(reply, turn.expectation)

        result = TurnEvalResult(
            case_id=case.case_id,
            turn_number=turn_number,
            user_text=turn.user_text,
            assistant_text=reply.text if reply is not None else None,
            error=error_text,
            passed=not failures,
            failures=failures,
            prompt_version=(
                reply.prompt_version if reply is not None else None
            ),
            history_messages=(
                reply.history_messages if reply is not None else None
            ),
            summary_updated=(
                reply.summary_updated if reply is not None else None
            ),
            has_summary=(
                reply.has_summary if reply is not None else None
            ),
        )
        results.append(result)

        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] Ход {turn_number}")
        print(f"User: {turn.user_text!r}")
        if reply is not None:
            print(f"Assistant: {reply.text}")
        if error_text is not None:
            print(f"Ошибка: {error_text}")
        for failure in failures:
            print(f"- {failure}")

    return results

def run_suite(chat_service: ChatService) -> list[TurnEvalResult]:
    results: list[TurnEvalResult] = []

    for case in CHAT_EVAL_CASES:
        results.extend(run_case(chat_service, case))

    return results