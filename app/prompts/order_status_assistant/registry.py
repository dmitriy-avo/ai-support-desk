from app.prompts.order_status_assistant.v1_0_0 import (
    ORDER_STATUS_ASSISTANT_PROMPT_V1_0_0,
)
from app.prompts.template import (
    PromptTemplate,
    PromptTemplateError,
)


ACTIVE_ORDER_STATUS_ASSISTANT_VERSION = "1.0.0"


_ORDER_STATUS_ASSISTANT_PROMPTS: dict[
    str,
    PromptTemplate,
] = {
    ORDER_STATUS_ASSISTANT_PROMPT_V1_0_0.version: (
        ORDER_STATUS_ASSISTANT_PROMPT_V1_0_0
    ),
}


def get_order_status_assistant_prompt(
    version: str | None = None,
) -> PromptTemplate:
    selected_version = (
        ACTIVE_ORDER_STATUS_ASSISTANT_VERSION
        if version is None
        else version.strip()
    )

    try:
        return _ORDER_STATUS_ASSISTANT_PROMPTS[
            selected_version
        ]
    except KeyError as error:
        available = ", ".join(
            _ORDER_STATUS_ASSISTANT_PROMPTS
        )
        raise PromptTemplateError(
            "Неизвестная версия order_status_assistant: "
            f"{selected_version}. Доступны: {available}."
        ) from error


def list_order_status_assistant_versions() -> tuple[str, ...]:
    return tuple(_ORDER_STATUS_ASSISTANT_PROMPTS)