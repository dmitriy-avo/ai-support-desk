from app.prompts.support_request_analysis.v1_0_0 import (
    SUPPORT_REQUEST_ANALYSIS_PROMPT_V1_0_0,
)
from app.prompts.template import PromptTemplate, PromptTemplateError


ACTIVE_SUPPORT_REQUEST_ANALYSIS_VERSION = "1.0.0"

_PROMPTS: dict[str, PromptTemplate] = {
    SUPPORT_REQUEST_ANALYSIS_PROMPT_V1_0_0.version: (
        SUPPORT_REQUEST_ANALYSIS_PROMPT_V1_0_0
    ),
}


def get_support_request_analysis_prompt(
    version: str | None = None,
) -> PromptTemplate:
    selected = (
        ACTIVE_SUPPORT_REQUEST_ANALYSIS_VERSION
        if version is None
        else version.strip()
    )

    try:
        return _PROMPTS[selected]
    except KeyError as error:
        available = ", ".join(_PROMPTS)
        raise PromptTemplateError(
            f"Неизвестная версия: {selected}. "
            f"Доступны: {available}."
        ) from error


def list_support_request_analysis_versions() -> tuple[str, ...]:
    return tuple(_PROMPTS)