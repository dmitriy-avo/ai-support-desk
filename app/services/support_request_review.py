from dataclasses import dataclass
from enum import StrEnum

from app.models.support_request import SupportCategory, Urgency
from app.services.support_request_analyzer import (
    SupportRequestResult,
)


MIN_AUTOMATIC_CONFIDENCE = 0.75


class ReviewReason(StrEnum):
    LOW_CONFIDENCE = "low_confidence"
    HIGH_URGENCY = "high_urgency"
    UNKNOWN_CATEGORY = "unknown_category"
    REPAIRED_RESULT = "repaired_result"


@dataclass(frozen=True)
class SupportRequestDecision:
    result: SupportRequestResult
    requires_human_review: bool
    review_reasons: tuple[ReviewReason, ...]


def decide_review(result: SupportRequestResult) -> SupportRequestDecision:
    analysis = result.analysis
    reasons: list[ReviewReason] = []

    if analysis.confidence < MIN_AUTOMATIC_CONFIDENCE:
        reasons.append(ReviewReason.LOW_CONFIDENCE)

    if analysis.urgency is Urgency.HIGH:
        reasons.append(ReviewReason.HIGH_URGENCY)

    if analysis.category is SupportCategory.OTHER:
        reasons.append(ReviewReason.UNKNOWN_CATEGORY)

    if result.repair_attempts > 0:
        reasons.append(ReviewReason.REPAIRED_RESULT)

    return SupportRequestDecision(
        result=result,
        requires_human_review=bool(reasons),
        review_reasons=tuple(reasons),
    )

