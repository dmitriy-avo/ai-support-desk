SUPPORT_REQUEST_SCHEMA_NAME = "support_request"

SUPPORT_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "description": "Категория обращения клиента.",
            "enum": [
                "technical_issue",
                "delivery_problem",
                "return_request",
                "payment_problem",
                "other",
            ],
        },
        "order_id": {
            "type": ["string", "null"],
            "description": (
                "Номер заказа из обращения или null, "
                "если клиент его не указал."
            ),
        },
        "product": {
            "type": ["string", "null"],
            "description": (
                "Название товара из обращения или null, "
                "если товар не указан."
            ),
        },
        "urgency": {
            "type": "string",
            "description": "Срочность обращения.",
            "enum": ["low", "normal", "high"],
        },
        "missing_fields": {
            "type": "array",
            "description": (
                "Сведения, которых не хватает для обработки."
            ),
            "items": {
                "type": "string",
                "enum": [
                    "order_id",
                    "product",
                    "problem_description",
                ],
            },
        },
    },
    "required": [
        "category",
        "order_id",
        "product",
        "urgency",
        "missing_fields",
    ],
    "additionalProperties": False,
}