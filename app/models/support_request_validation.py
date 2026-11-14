import re

from app.models.support_request import (
    MissingField,
    SupportRequestAnalysis,
)


class SupportRequestSemanticError(ValueError):
    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__(" ".join(issues))


def validate_support_request(analysis: SupportRequestAnalysis, user_text: str) -> None:
    issues: list[str] = []

    fields_to_check = (
        (
            "order_id",
            analysis.order_id,
            MissingField.ORDER_ID,
        ),
        (
            "product",
            analysis.product,
            MissingField.PRODUCT,
        ),
        (
            "problem_description",
            analysis.problem_description,
            MissingField.PROBLEM_DESCRIPTION,
        ),
    )

    for field_name, value, missing_field in fields_to_check:
        is_marked_missing = (
                missing_field in analysis.missing_fields
        )

        if value is None and not is_marked_missing:
            issues.append(
                f"{field_name} равен null, "
                "но не указан в missing_fields."
            )

        if value is not None and is_marked_missing:
            issues.append(
                f"{field_name} заполнен, "
                "но указан в missing_fields."
            )

    if analysis.order_id is not None:
        order_ids = set(
            re.findall(r"\b\d+\b", user_text)
        )
        if analysis.order_id not in order_ids:
            issues.append(
                "order_id отсутствует в исходном обращении."
            )

    if issues:
        raise SupportRequestSemanticError(tuple(issues))
