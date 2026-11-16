from app.prompts.order_status_assistant.registry import (
    ACTIVE_ORDER_STATUS_ASSISTANT_VERSION,
    get_order_status_assistant_prompt,
    list_order_status_assistant_versions,
)


ORDER_STATUS_ASSISTANT_PROMPT = (
    get_order_status_assistant_prompt()
)


__all__ = [
    "ACTIVE_ORDER_STATUS_ASSISTANT_VERSION",
    "ORDER_STATUS_ASSISTANT_PROMPT",
    "get_order_status_assistant_prompt",
    "list_order_status_assistant_versions",
]