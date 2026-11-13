from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SupportCategory(StrEnum):
    TECHNICAL_ISSUE = "technical_issue"
    DELIVERY_PROBLEM = "delivery_problem"
    RETURN_REQUEST = "return_request"
    PAYMENT_PROBLEM = "payment_problem"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class MissingField(StrEnum):
    ORDER_ID = "order_id"
    PRODUCT = "product"
    PROBLEM_DESCRIPTION = "problem_description"


class SupportRequestAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: SupportCategory
    order_id: str | None = Field(description="Номер заказа или null, если он не указан.")
    product: str | None = Field(description="Название товара или null, если оно не указано.")
    urgency: Urgency
    missing_fields: list[MissingField] = Field(
        description="Сведения, которых не хватает для обработки обращения.")

    @field_validator("order_id")
    @classmethod
    def validate_order_id(cls, value: str | None) -> str | None:
        if value is None:
            return None

        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "Номер заказа не должен быть пустой строкой."
            )
        if not cleaned.isdigit():
            raise ValueError(
                "Номер заказа должен содержать только цифры."
            )
        return cleaned

    @field_validator("missing_fields")
    @classmethod
    def validate_missing_fields(
            cls,
            value: list[MissingField],
    ) -> list[MissingField]:
        if len(value) != len(set(value)):
            raise ValueError(
                "Список missing_fields не должен содержать повторы."
            )
        return value
