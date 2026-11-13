import re

from app.models.support_request import (
    MissingField,
    SupportRequestAnalysis,
)


class SupportRequestSemanticError(ValueError):
    def __init__(self, issues: tuple[str, ...]) -> None:
        self.issues = issues
        super().__init__(" ".join(issues))


def validate_support_request(
        analysis: SupportRequestAnalysis,
        user_text: str,
) -> None:
    issues: list[str] = []

    order_id_is_missing = (
            MissingField.ORDER_ID in analysis.missing_fields
    )
    if analysis.order_id is None and not order_id_is_missing:
        issues.append(
            "order_id равен null, но не указан "
            "в missing_fields."
        )
    if analysis.order_id is not None and order_id_is_missing:
        issues.append(
            "order_id заполнен, но указан "
            "в missing_fields."
        )

    product_is_missing = (
            MissingField.PRODUCT in analysis.missing_fields
    )
    if analysis.product is None and not product_is_missing:
        issues.append(
            "product равен null, но не указан "
            "в missing_fields."
        )
    if analysis.product is not None and product_is_missing:
        issues.append(
            "product заполнен, но указан "
            "в missing_fields."
        )

    if analysis.order_id is not None:
        order_ids = set(re.findall(r"\b\d+\b", user_text))
        if analysis.order_id not in order_ids:
            issues.append(
                "order_id отсутствует в исходном обращении."
            )

    if issues:
        raise SupportRequestSemanticError(tuple(issues))
