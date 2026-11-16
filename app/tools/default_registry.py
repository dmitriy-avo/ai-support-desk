from app.tools.order_status import ORDER_STATUS_TOOL
from app.tools.registry import ToolRegistry
from app.tools.return_policy import RETURN_POLICY_TOOL


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    for tool in (ORDER_STATUS_TOOL, RETURN_POLICY_TOOL):
        registry.register(tool)

    return registry