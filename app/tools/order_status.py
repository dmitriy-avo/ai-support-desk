from enum import StrEnum
from app.tools.registry import RegisteredTool

from pydantic import BaseModel, ConfigDict, Field


GET_ORDER_STATUS_TOOL_NAME = "get_order_status"


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    IN_TRANSIT = "in_transit"
    READY_FOR_PICKUP = "ready_for_pickup"
    DELIVERED = "delivered"


class GetOrderStatusArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(
        pattern=r"^\d{4}$",
        description="Номер заказа из четырёх цифр.",
    )


class OrderStatusResult(BaseModel):
    order_id: str
    found: bool
    status: OrderStatus | None
    message: str


ORDER_DATA: dict[str, OrderStatus] = {
    "8041": OrderStatus.IN_TRANSIT,
    "7718": OrderStatus.DELIVERED,
    "9912": OrderStatus.READY_FOR_PICKUP,
}


def get_order_status(
    arguments: GetOrderStatusArguments,
) -> OrderStatusResult:
    status = ORDER_DATA.get(arguments.order_id)

    if status is None:
        return OrderStatusResult(
            order_id=arguments.order_id,
            found=False,
            status=None,
            message="Заказ с таким номером не найден.",
        )

    return OrderStatusResult(
        order_id=arguments.order_id,
        found=True,
        status=status,
        message="Статус заказа успешно получен.",
    )


ORDER_STATUS_TOOL = RegisteredTool[GetOrderStatusArguments, OrderStatusResult](
    name=GET_ORDER_STATUS_TOOL_NAME,
    description=(
        "Возвращает актуальный статус заказа "
        "по его четырёхзначному номеру. "
        "Используй инструмент только для вопроса "
        "о статусе или местонахождении заказа."
    ),
    arguments_model=GetOrderStatusArguments,
    handler=get_order_status,
    timeout_seconds=2.0,
)