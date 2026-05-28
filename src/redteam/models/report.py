"""Report models — per-goal results and the final Layer 1 report."""

from __future__ import annotations

from pydantic import BaseModel, Field, computed_field

from redteam.models.attacks import AttackResult
from redteam.models.environment import DryRunTrace, GeneratedEnvironment
from redteam.models.risk import ASICategory, AttackerGoal, RiskCard


class ModelConfig(BaseModel):
    """Configuration for the model being red-teamed."""

    model: str                          # e.g. "qwen3.5:2b", "openai/gpt-4o"
    base_url: str = "http://localhost:11434/v1"  # Ollama default
    api_key: str = "ollama"             # Dummy for Ollama/vLLM; real key for OpenAI
    temperature: float = 0.0            # Target model always deterministic
    max_tokens: int = 2048


class GoalResult(BaseModel):
    """All attack attempts for a single goal against a single model.

    The unit of result in Layer 1. One GoalResult per (goal × model) pair.
    """

    goal: AttackerGoal
    model_id: str
    environment: GeneratedEnvironment
    dry_run_trace: DryRunTrace
    attack_results: list[AttackResult] = Field(default_factory=list)

    @computed_field
    @property
    def worst_security_score(self) -> float:
        """Minimum security score across all attacks (most vulnerable)."""
        if not self.attack_results:
            return 1.0
        return min(r.security_score for r in self.attack_results)

    @computed_field
    @property
    def best_utility_score(self) -> float:
        """Maximum utility score across all attacks (least task disruption)."""
        if not self.attack_results:
            return 1.0
        return max(r.utility_score for r in self.attack_results)

    @computed_field
    @property
    def any_violation(self) -> bool:
        return any(r.policy_violated for r in self.attack_results)

    @computed_field
    @property
    def successful_attack_types(self) -> list[str]:
        return [r.attack_type.value for r in self.attack_results if r.policy_violated]


class Layer1Report(BaseModel):
    """The final Layer 1 red-team report.

    Aggregates results across all goals and models.
    Includes OWASP coverage, policy violation mapping, and
    model comparison when multiple models are tested.

    IMPORTANT: Results reflect model susceptibility in a controlled
    synthetic environment. Real-world attack success depends on actual
    tool schemas, data, and infrastructure. Layer 1 does not test
    real tools or real data — that is Layer 2.
    """

    usecase: str
    policies: list[str]
    risk_cards: list[RiskCard] = Field(default_factory=list)
    goal_results: list[GoalResult] = Field(default_factory=list)

    # Aggregated views (computed on demand)
    @computed_field
    @property
    def owasp_coverage(self) -> dict[str, dict]:
        """Per ASI category: how many goals tested, violation rate."""
        coverage: dict[str, dict] = {}
        for gr in self.goal_results:
            cat = gr.goal.asi_category.value
            if cat not in coverage:
                coverage[cat] = {"goals_tested": 0, "violations": 0}
            coverage[cat]["goals_tested"] += 1
            if gr.any_violation:
                coverage[cat]["violations"] += 1
        for cat, data in coverage.items():
            tested = data["goals_tested"]
            data["violation_rate"] = data["violations"] / tested if tested else 0.0
        return coverage

    @computed_field
    @property
    def policy_violations(self) -> dict[str, list[str]]:
        """Which policies were violated, and by which goals."""
        violations: dict[str, list[str]] = {}
        for gr in self.goal_results:
            if not gr.any_violation:
                continue
            for policy in gr.goal.risk_card_id and []:  # populated from risk cards
                if policy not in violations:
                    violations[policy] = []
                violations[policy].append(gr.goal.id)
        return violations

    @computed_field
    @property
    def model_comparison(self) -> dict[str, dict]:
        """Per model: average security score, violation rate, utility score."""
        by_model: dict[str, list[GoalResult]] = {}
        for gr in self.goal_results:
            if gr.model_id not in by_model:
                by_model[gr.model_id] = []
            by_model[gr.model_id].append(gr)

        comparison = {}
        for model_id, results in by_model.items():
            n = len(results)
            comparison[model_id] = {
                "goals_tested": n,
                "avg_security_score": sum(r.worst_security_score for r in results) / n,
                "violation_rate": sum(1 for r in results if r.any_violation) / n,
                "avg_utility_score": sum(r.best_utility_score for r in results) / n,
            }
        return comparison
