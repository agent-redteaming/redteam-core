"""Risk models — RiskCard, triage types, and AttackerGoal.

RiskCard structure ported from agent-policy-redteam/models.py.
AttackerGoal and ASI mapping are new — the explicit bridge between
a risk description and a specific, measurable, testable attack objective.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# RiskCard components (from agent-policy-redteam's models.py)
# ---------------------------------------------------------------------------


class RiskSource(BaseModel):
    description: str
    likelihood: Literal["low", "medium", "high", "critical"]


class RiskConsequence(BaseModel):
    description: str
    severity: Literal["low", "medium", "high", "critical"]


class RiskImpact(BaseModel):
    description: str
    affected_stakeholders: list[str]
    harm_type: str


class RiskControl(BaseModel):
    type: Literal["detect", "evaluate", "mitigate", "eliminate"]
    description: str


class RiskCard(BaseModel):
    """Structured risk assessment artifact.

    Provided by the user (from risk-landscaper or manually authored)
    or generated from usecase + policy statements.
    """

    id: str
    risk_source: RiskSource
    risk_consequence: RiskConsequence
    risk_impact: RiskImpact
    risk_controls: list[RiskControl]
    materialization_conditions: str
    policy_references: list[str]
    framework_references: list[str]


# ---------------------------------------------------------------------------
# Triage types (from agent-policy-redteam's triage.py — adapted as enums)
# ---------------------------------------------------------------------------


class RiskType(str, Enum):
    """What kind of policy violation this risk represents."""

    DATA_EXFILTRATION = "data_exfiltration"
    UNAUTHORIZED_ACTION = "unauthorized_action"
    DATA_CORRUPTION = "data_corruption"
    MEMORY_POISONING = "memory_poisoning"  # ASI-06 goals
    OTHER = "other"


class EnforcementLevel(str, Enum):
    """Who enforces this policy — agent behaviour or sandbox infrastructure.

    Layer 1 only processes AGENT-level risks. SANDBOX risks require
    infrastructure enforcement (network policies, filesystem isolation)
    which is Layer 3 territory.
    """

    AGENT = "agent"
    SANDBOX = "sandbox"


class TriagedRisk(BaseModel):
    risk_card: RiskCard
    enforcement_level: EnforcementLevel
    risk_type: RiskType


# ---------------------------------------------------------------------------
# OWASP ASI categories
# ---------------------------------------------------------------------------


class ASICategory(str, Enum):
    """OWASP Top 10 for Agentic Applications (December 2025).

    These map to architectural components that fail, not to attack techniques.
    A single RiskCard can generate goals across multiple ASI categories.
    """

    ASI_01 = "ASI-01"  # Agent Goal Hijack
    ASI_02 = "ASI-02"  # Tool Misuse and Exploitation
    ASI_03 = "ASI-03"  # Identity and Privilege Abuse
    ASI_04 = "ASI-04"  # Agentic Supply Chain Vulnerabilities
    ASI_05 = "ASI-05"  # Unexpected Code Execution
    ASI_06 = "ASI-06"  # Memory and Context Poisoning
    ASI_07 = "ASI-07"  # Insecure Inter-Agent Communication
    ASI_08 = "ASI-08"  # Cascading Failures
    ASI_09 = "ASI-09"  # Human-Agent Trust Exploitation
    ASI_10 = "ASI-10"  # Rogue Agents


# ---------------------------------------------------------------------------
# Attack channels
# ---------------------------------------------------------------------------


class AttackChannel(str, Enum):
    """Where the adversarial input enters the agent.

    Different channels require different attack implementations but all
    run against the same synthetic environment — the environment provides
    the tools and the observability regardless of which channel is used.
    """

    TOOL_RESPONSE = "tool_response"       # Injection: payload in tool result
    USER_MESSAGE = "user_message"         # PAIR/T-MAP: adversarial user prompt
    CONVERSATION = "conversation"         # Multi-turn: escalating across turns
    TOOL_DESCRIPTION = "tool_description" # Poisoned runtime: hidden in schema
    MEMORY = "memory"                     # MINJA: poisoned few-shot memory


# ---------------------------------------------------------------------------
# AttackerGoal — the unit of work for the per-goal pipeline
# ---------------------------------------------------------------------------


class AttackerGoal(BaseModel):
    """A specific, measurable attacker objective derived from a RiskCard.

    One RiskCard generates 2-3 goals. Each goal is specific enough to:
      - Generate a purpose-built synthetic environment
      - Define a concrete success criterion
      - Map to an OWASP ASI category
      - Select which attack channels are applicable

    This is the key bridge between threat model (RiskCard) and test case.
    """

    id: str                                  # e.g. "RC-001-G1"
    risk_card_id: str
    attacker_objective: str                  # "Make agent send confidential records to an external address"
    success_criteria: str                    # "communication tool called with sensitive data, external recipient"
    asi_category: ASICategory
    risk_type: RiskType
    severity: Literal["critical", "high", "medium", "low"]
    attack_channels: list[AttackChannel]
