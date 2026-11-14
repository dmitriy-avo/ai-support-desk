from app.prompts.support_request_analysis.registry import (
    ACTIVE_SUPPORT_REQUEST_ANALYSIS_VERSION,
    get_support_request_analysis_prompt,
    list_support_request_analysis_versions,
)


SUPPORT_REQUEST_ANALYSIS_PROMPT = get_support_request_analysis_prompt()

__all__ = [
    "ACTIVE_SUPPORT_REQUEST_ANALYSIS_VERSION",
    "SUPPORT_REQUEST_ANALYSIS_PROMPT",
    "get_support_request_analysis_prompt",
    "list_support_request_analysis_versions",
]
