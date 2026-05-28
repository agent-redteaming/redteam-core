"""Risk triage — classify RiskCards by enforcement level and risk type.

Ported verbatim from agent-policy-redteam/triage.py.

Enforcement level:
  agent:   risk is prevented by agent behaviour/output filtering
  sandbox: risk is prevented by infrastructure (network, filesystem, containers)
  Layer 1 only processes agent-level risks.

Risk type drives which environment template is selected:
  data_exfiltration:   sensitive data leaks to external party
  unauthorized_action: agent performs actions beyond its authority
  data_corruption:     agent tampers with integrity-critical data
  memory_poisoning:    agent's memory store is poisoned (ASI-06 goals)
  other:               fallback
"""

from __future__ import annotations

from redteam.models.risk import EnforcementLevel, RiskCard, RiskType, TriagedRisk


def triage_risk(risk_card: RiskCard) -> TriagedRisk:
    """Classify a RiskCard by enforcement level and risk type."""

    # --- Enforcement level ---
    # Sandbox keywords in control descriptions indicate infrastructure enforcement
    sandbox_keywords = [
        "network", "filesystem", "process", "credential",
        "sandbox", "container", "isolation",
    ]
    controls_text = " ".join(c.description.lower() for c in risk_card.risk_controls)
    enforcement_level = (
        EnforcementLevel.SANDBOX
        if any(kw in controls_text for kw in sandbox_keywords)
        else EnforcementLevel.AGENT
    )

    # --- Risk type ---
    risk_text = (
        risk_card.risk_source.description.lower()
        + " "
        + risk_card.risk_consequence.description.lower()
    )

    exfiltration_keywords = ["exfiltrat", "leak", "expos", "sensitive data", "disclosure", "pii"]
    unauthorized_keywords = ["privilege", "unauthorized", "permission", "escalat", "access control"]
    corruption_keywords = ["integrity", "tamper", "corrupt", "modif", "alter"]
    memory_keywords = ["memory", "poisoning", "retrieval", "rag", "few-shot", "in-context"]

    if any(kw in risk_text for kw in memory_keywords):
        risk_type = RiskType.MEMORY_POISONING
    elif any(kw in risk_text for kw in exfiltration_keywords):
        risk_type = RiskType.DATA_EXFILTRATION
    elif any(kw in risk_text for kw in unauthorized_keywords):
        risk_type = RiskType.UNAUTHORIZED_ACTION
    elif any(kw in risk_text for kw in corruption_keywords):
        risk_type = RiskType.DATA_CORRUPTION
    else:
        risk_type = RiskType.OTHER

    return TriagedRisk(
        risk_card=risk_card,
        enforcement_level=enforcement_level,
        risk_type=risk_type,
    )


def filter_agent_level(risks: list[TriagedRisk]) -> list[TriagedRisk]:
    """Keep only agent-level risks. Sandbox-level risks deferred to Layer 3."""
    return [r for r in risks if r.enforcement_level == EnforcementLevel.AGENT]
