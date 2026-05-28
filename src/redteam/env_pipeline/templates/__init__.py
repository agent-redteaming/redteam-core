"""Environment generation templates — one per risk type."""

from redteam.env_pipeline.templates.base import COMMON_RULES
from redteam.env_pipeline.templates.data_corruption import TEMPLATE_PROMPT as CORRUPTION_PROMPT
from redteam.env_pipeline.templates.data_exfiltration import TEMPLATE_PROMPT as EXFIL_PROMPT
from redteam.env_pipeline.templates.memory_poisoning import TEMPLATE_PROMPT as MEMORY_PROMPT
from redteam.env_pipeline.templates.unauthorized_action import TEMPLATE_PROMPT as UNAUTH_PROMPT
from redteam.models.risk import RiskType

TEMPLATES: dict[RiskType, str] = {
    RiskType.DATA_EXFILTRATION: EXFIL_PROMPT,
    RiskType.UNAUTHORIZED_ACTION: UNAUTH_PROMPT,
    RiskType.DATA_CORRUPTION: CORRUPTION_PROMPT,
    RiskType.MEMORY_POISONING: MEMORY_PROMPT,
    RiskType.OTHER: EXFIL_PROMPT,  # default to exfil template for unknown types
}

__all__ = ["COMMON_RULES", "TEMPLATES"]
