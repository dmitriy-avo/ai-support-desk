from app.models.support_request import SupportRequestAnalysis

SUPPORT_REQUEST_SCHEMA_NAME = "support_request"
SUPPORT_REQUEST_SCHEMA = SupportRequestAnalysis.model_json_schema()
