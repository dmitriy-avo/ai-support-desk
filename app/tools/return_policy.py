from pydantic import BaseModel, ConfigDict, Field

from app.tools.registry import RegisteredTool


GET_RETURN_POLICY_TOOL_NAME = "get_return_policy"


class GetReturnPolicyArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    product: str = Field(
        min_length=2,
        max_length=100,
        description=(
            "Название товара без номера заказа, "
            "например кофемашина или наушники."
        ),
    )


class ReturnPolicyResult(BaseModel):
    product: str
    found: bool
    return_days: int | None
    conditions: str


RETURN_POLICY_DATA: dict[str, tuple[int, str]] = {
    "кофемашина": (
        14,
        "Товар должен сохранить комплектность "
        "и не иметь следов использования.",
    ),
    "наушники": (
        7,
        "Возврат возможен при сохранении упаковки "
        "и товарного вида.",
    ),
    "клавиатура": (
        14,
        "Нужны комплектность и отсутствие "
        "механических повреждений.",
    ),
}


def get_return_policy(arguments: GetReturnPolicyArguments) -> ReturnPolicyResult:
    product = arguments.product.strip().lower()
    policy = RETURN_POLICY_DATA.get(product)

    if policy is None:
        return ReturnPolicyResult(
            product=product,
            found=False,
            return_days=None,
            conditions="Отдельная политика для товара не найдена.",
        )

    return_days, conditions = policy
    return ReturnPolicyResult(
        product=product,
        found=True,
        return_days=return_days,
        conditions=conditions,
    )


RETURN_POLICY_TOOL = RegisteredTool[GetReturnPolicyArguments, ReturnPolicyResult](
    name=GET_RETURN_POLICY_TOOL_NAME,
    description=(
        "Возвращает срок и условия возврата "
        "для указанного товара. "
        "Используй инструмент для вопросов "
        "о возможности, сроке или условиях возврата."
    ),
    arguments_model=GetReturnPolicyArguments,
    handler=get_return_policy,
    timeout_seconds=2.0,
)