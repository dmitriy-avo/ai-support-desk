from pydantic import ValidationError

from app.config import Settings
from app.console import run_console
from app.llm.client import LLMClient
from app.services.chat_service import ChatService
from app.services.support_request_analyzer import SupportRequestAnalyzer
from app.chat_history import (
    SlidingWindowHistory,
    TokenCounter,
)
from app.services.conversation_summary_service import (
    ConversationSummaryService,
)
from app.services.tool_loop import BoundedToolLoop
from app.tools.default_registry import build_default_tool_registry
from app.tools.execution import ToolExecutor

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
    tool_registry = build_default_tool_registry()
    tool_executor = ToolExecutor(tool_registry)
    tool_loop = BoundedToolLoop(
        client=llm_client,
        registry=tool_registry,
        executor=tool_executor,
        max_steps=settings.tool_max_steps,
    )
    token_counter = TokenCounter()
    history_policy = SlidingWindowHistory(
        token_counter=token_counter,
        input_token_budget=settings.chat_input_token_budget,
    )

    summary_service = ConversationSummaryService(llm_client)
    chat_service = ChatService(
        tool_loop=tool_loop,
        history_policy=history_policy,
        summary_service=summary_service,
    )
    request_analyzer = SupportRequestAnalyzer(llm_client)


    try:
        run_console(
            chat_service,
            request_analyzer,
            app_env=settings.app_env,
            configured_model=settings.model,
        )
    finally:
        tool_executor.close()
        llm_client.close()


if __name__ == "__main__":
    main()