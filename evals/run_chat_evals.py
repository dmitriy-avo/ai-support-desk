from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from pydantic import ValidationError

from app.chat_history import SlidingWindowHistory, TokenCounter
from app.chat_service import ChatService
from app.config import Settings
from app.conversation_summary_service import ConversationSummaryService
from app.llm.client import LLMClient
from app.prompts.support_chat import get_support_chat_prompt
from app.prompts.template import PromptTemplate
from evals.chat_checks import run_suite
from evals.chat_models import TurnEvalResult


PROMPT_VERSIONS = ("1.0.0", "1.1.0")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def build_chat_service(
    settings: Settings,
    llm_client: LLMClient,
    prompt: PromptTemplate,
) -> ChatService:
    token_counter = TokenCounter()
    history_policy = SlidingWindowHistory(
        token_counter=token_counter,
        input_token_budget=settings.chat_input_token_budget,
    )
    summary_service = ConversationSummaryService(llm_client)

    return ChatService(
        llm_client=llm_client,
        history_policy=history_policy,
        summary_service=summary_service,
        prompt=prompt,
    )


def save_report(
    settings: Settings,
    prompt: PromptTemplate,
    results: list[TurnEvalResult],
) -> Path:
    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "created_at": created_at.isoformat(),
        "model": settings.model,
        "temperature": settings.temperature,
        "max_output_tokens": settings.max_output_tokens,
        "chat_input_token_budget": settings.chat_input_token_budget,
        "prompt": {
            "prompt_id": prompt.prompt_id,
            "version": prompt.version,
            "body": prompt.body,
        },
        "results": [asdict(result) for result in results],
    }

    report_path = RESULTS_DIR / (
        f"{prompt.prompt_id}-{prompt.version}-{timestamp}.json"
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def count_passed_cases(
    results: list[TurnEvalResult],
) -> tuple[int, int]:
    case_statuses: dict[str, bool] = {}

    for result in results:
        previous_status = case_statuses.get(result.case_id, True)
        case_statuses[result.case_id] = previous_status and result.passed

    passed = sum(case_statuses.values())
    return passed, len(case_statuses)


def print_summary(
    prompt: PromptTemplate,
    results: list[TurnEvalResult],
    report_path: Path,
) -> None:
    passed, total = count_passed_cases(results)
    print(f"\nИтог для {prompt.prompt_id}@{prompt.version}: {passed}/{total}")
    print(f"Отчет: {report_path}")


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
    summaries: dict[str, tuple[int, int]] = {}

    try:
        for version in PROMPT_VERSIONS:
            prompt = get_support_chat_prompt(version)
            chat_service = build_chat_service(
                settings=settings,
                llm_client=llm_client,
                prompt=prompt,
            )

            print(f"\n=== {prompt.prompt_id}@{prompt.version} ===")
            results = run_suite(chat_service)
            report_path = save_report(settings, prompt, results)
            print_summary(prompt, results, report_path)
            summaries[version] = count_passed_cases(results)
    finally:
        llm_client.close()

    print("\nСравнение автоматических проверок:")
    for version, (passed, total) in summaries.items():
        print(f"- {version}: {passed}/{total} сценариев")


if __name__ == "__main__":
    main()


