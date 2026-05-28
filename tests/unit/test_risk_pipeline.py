"""Unit tests for risk pipeline — no LLM calls."""

from __future__ import annotations

import pytest

from redteam.models.risk import (
    ASICategory,
    AttackChannel,
    EnforcementLevel,
    RiskCard,
    RiskConsequence,
    RiskControl,
    RiskImpact,
    RiskSource,
    RiskType,
    TriagedRisk,
)
from redteam.risk_pipeline.triage import filter_agent_level, triage_risk


def make_card(
    id: str = "RC-001",
    risk_source_desc: str = "Agent could exfiltrate sensitive PII",
    risk_consequence_desc: str = "Unauthorized disclosure of employee data",
    control_descs: list[str] | None = None,
) -> RiskCard:
    controls = [
        RiskControl(type="detect", description=desc)
        for desc in (control_descs or ["Implement logging of all data access patterns"])
    ]
    return RiskCard(
        id=id,
        risk_source=RiskSource(description=risk_source_desc, likelihood="high"),
        risk_consequence=RiskConsequence(description=risk_consequence_desc, severity="critical"),
        risk_impact=RiskImpact(
            description="Employees face privacy violations",
            affected_stakeholders=["employees"],
            harm_type="Privacy violation",
        ),
        risk_controls=controls,
        materialization_conditions="Agent reads records containing injected instructions",
        policy_references=["GDPR Article 5"],
        framework_references=["OWASP LLM01"],
    )


# ---------------------------------------------------------------------------
# triage_risk — enforcement level
# ---------------------------------------------------------------------------


def test_triage_agent_level_default():
    card = make_card(control_descs=["Implement output filtering and sanitization"])
    result = triage_risk(card)
    assert result.enforcement_level == EnforcementLevel.AGENT


def test_triage_sandbox_level_network():
    card = make_card(control_descs=["Restrict network access at the container level"])
    result = triage_risk(card)
    assert result.enforcement_level == EnforcementLevel.SANDBOX


def test_triage_sandbox_level_filesystem():
    card = make_card(control_descs=["Use filesystem isolation to prevent unauthorised writes"])
    result = triage_risk(card)
    assert result.enforcement_level == EnforcementLevel.SANDBOX


def test_triage_sandbox_level_container():
    card = make_card(control_descs=["Deploy in a sandboxed container environment"])
    result = triage_risk(card)
    assert result.enforcement_level == EnforcementLevel.SANDBOX


# ---------------------------------------------------------------------------
# triage_risk — risk type
# ---------------------------------------------------------------------------


def test_triage_data_exfiltration():
    card = make_card(
        risk_source_desc="Agent could exfiltrate sensitive employee PII",
        risk_consequence_desc="Unauthorized disclosure of personal data",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.DATA_EXFILTRATION


def test_triage_data_exfiltration_leak():
    card = make_card(
        risk_source_desc="Sensitive data could leak to external parties",
        risk_consequence_desc="Privacy violation and data exposure",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.DATA_EXFILTRATION


def test_triage_unauthorized_action():
    card = make_card(
        risk_source_desc="Agent could escalate privileges without authorization",
        risk_consequence_desc="Unauthorized access to admin functions",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.UNAUTHORIZED_ACTION


def test_triage_unauthorized_permission():
    card = make_card(
        risk_source_desc="Permission escalation via manipulated tool responses",
        risk_consequence_desc="Breach of access control policies",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.UNAUTHORIZED_ACTION


def test_triage_data_corruption():
    card = make_card(
        risk_source_desc="Agent could tamper with financial records",
        risk_consequence_desc="Data integrity violations in invoice system",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.DATA_CORRUPTION


def test_triage_memory_poisoning():
    card = make_card(
        risk_source_desc="Agent's memory retrieval could be poisoned with adversarial examples",
        risk_consequence_desc="Agent follows malicious in-context examples from poisoned memory",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.MEMORY_POISONING


def test_triage_other_fallback():
    card = make_card(
        risk_source_desc="Unusual agent behavior causing reputational damage",
        risk_consequence_desc="Brand damage and user trust erosion",
    )
    result = triage_risk(card)
    assert result.risk_type == RiskType.OTHER


# ---------------------------------------------------------------------------
# filter_agent_level
# ---------------------------------------------------------------------------


def test_filter_agent_level():
    agent_card = make_card(
        "RC-001",
        control_descs=["Output filtering and monitoring"],
    )
    sandbox_card = make_card(
        "RC-002",
        control_descs=["Network isolation at the container level"],
    )

    risks = [triage_risk(agent_card), triage_risk(sandbox_card)]
    filtered = filter_agent_level(risks)

    assert len(filtered) == 1
    assert filtered[0].risk_card.id == "RC-001"
    assert filtered[0].enforcement_level == EnforcementLevel.AGENT


def test_filter_agent_level_empty():
    sandbox_card = make_card(control_descs=["Sandbox isolation required"])
    risks = [triage_risk(sandbox_card)]
    filtered = filter_agent_level(risks)
    assert filtered == []


def test_filter_agent_level_all_agent():
    cards = [make_card(f"RC-{i:03d}") for i in range(3)]
    risks = [triage_risk(c) for c in cards]
    filtered = filter_agent_level(risks)
    assert len(filtered) == 3


# ---------------------------------------------------------------------------
# TriagedRisk serialisation
# ---------------------------------------------------------------------------


def test_triaged_risk_roundtrip():
    card = make_card()
    triaged = triage_risk(card)
    restored = TriagedRisk.model_validate_json(triaged.model_dump_json())
    assert restored.risk_card.id == card.id
    assert restored.enforcement_level == triaged.enforcement_level
    assert restored.risk_type == triaged.risk_type
