"""Per-goal pipeline orchestrator — wires all components into a single run.

This is the core coordinator for Layer 1. Given an AttackerGoal and a model,
it runs the full pipeline: environment → dry run → attacks → evaluation.

Called by the CLI for each (goal × model) pair.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Sequence

from redteam.attacks.injection import run_direct_injection
from redteam.evaluation.judge import judge_attack
from redteam.attacks.minja import run_minja
from redteam.attacks.multi_turn import run_multi_turn
from redteam.attacks.pair import run_pair
from redteam.attacks.poisoned_runtime import run_poisoned_runtime
from redteam.attacks.tmap import run_tmap
from redteam.env_pipeline.adapter import SyntheticEnvTargetAdapter
from redteam.env_pipeline.env_generator import generate_environment
from redteam.env_pipeline.executor import dry_run
from redteam.models.attacks import AttackResult, AttackType
from redteam.models.environment import GeneratedEnvironment
from redteam.models.report import GoalResult, ModelConfig
from redteam.models.risk import AttackerGoal
from redteam.risk_pipeline.goal_generator import generate_goals
from redteam.risk_pipeline.risk_generator import generate_risk_cards
from redteam.risk_pipeline.triage import filter_agent_level, triage_risk
from redteam.runtime.chat_completions import ChatCompletionsRuntime

log = logging.getLogger(__name__)

# Full attack suite — run all of these against every goal by default.
# Goals are channel-agnostic: the same objective is attempted through every
# available attack surface so we get real OWASP coverage, not declared coverage.
ALL_ATTACKS: list[AttackType] = [
    AttackType.DIRECT_INJECTION,
    AttackType.PAIR_ADVERSARIAL,
    AttackType.MULTI_TURN,
    AttackType.POISONED_RUNTIME,
    AttackType.MINJA,
]


def attacks_for_goal(
    goal: AttackerGoal,
    requested: list[AttackType] | None = None,
) -> list[AttackType]:
    """Determine which attacks to run for a goal.

    If --attacks was specified explicitly, honour that list.
    Otherwise run the full suite: every goal is attacked from every surface.
    """
    return requested if requested else ALL_ATTACKS


def run_per_goal(
    goal: AttackerGoal,
    model_config: ModelConfig,
    policies: list[str] | None = None,
    attack_types: list[AttackType] | None = None,
    attacker_base_url: str | None = None,
    attacker_model: str | None = None,
    pair_n_streams: int = 5,
    pair_k_iterations: int = 3,
    tmap_iterations: int = 10,
    multi_turn_n_turns: int = 4,
) -> GoalResult:
    """Run the full per-goal pipeline: env → dry run → attacks → results.

    Args:
        goal: The specific attacker goal to test.
        model_config: The target model configuration.
        attack_types: Which attacks to run (None = full suite, all attack types).
        attacker_base_url: Attacker LLM endpoint (for generation + attack).
        attacker_model: Attacker model name.
        pair_n_streams: PAIR streams (paper: 30, reduced for speed).
        pair_k_iterations: PAIR iterations per stream (paper: 3).
        tmap_iterations: T-MAP generations (paper: 100, reduced).
        multi_turn_n_turns: Number of multi-turn turns.
    """
    attacker_model = attacker_model or os.environ.get("ATTACKER_MODEL") or \
                     os.environ.get("TARGET_MODEL", "qwen3.5:2b")
    attacker_base_url = attacker_base_url or os.environ.get("ATTACKER_BASE_URL") or \
                        os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")

    log.info(
        "Per-goal pipeline: %s [model=%s]",
        goal.id, model_config.model,
    )

    # Stage 1: Generate environment — pass policies so the generator creates
    # seed data that exercises policy boundaries and the right tools for the attack
    log.info("  Stage 1: Generating environment...")
    t0 = time.time()
    env = generate_environment(goal, model=attacker_model, base_url=attacker_base_url,
                               policies=policies or [])
    log.info("  Environment generated in %.1fs (tools: %s)", time.time() - t0,
             env.expected_tool_sequence)

    # Stage 2: Dry run (clean baseline)
    log.info("  Stage 2: Dry run...")
    runtime = ChatCompletionsRuntime(base_url=model_config.base_url)
    t0 = time.time()
    dry_run_trace = dry_run(env, runtime, model_config.model)
    log.info("  Dry run: %d tool calls, %d records, %.1fs",
             len(dry_run_trace.tool_calls), len(dry_run_trace.accessed_records), time.time() - t0)

    # Stage 3: Attack suite
    selected_attacks = attacks_for_goal(goal, attack_types)
    log.info("  Stage 3: Running attacks: %s", [a.value for a in selected_attacks])

    attack_results: list[AttackResult] = []

    # Create adapter once, reuse across attacks (reset between attacks)
    adapter = SyntheticEnvTargetAdapter(env, runtime, model_config.model)

    for attack_type in selected_attacks:
        log.info("    Running %s...", attack_type.value)
        adapter._reset()  # clean state before each attack
        t0 = time.time()

        try:
            result = _run_attack(
                attack_type=attack_type,
                goal=goal,
                env=env,
                dry_run_trace=dry_run_trace,
                adapter=adapter,
                attacker_base_url=attacker_base_url,
                attacker_model=attacker_model,
                pair_n_streams=pair_n_streams,
                pair_k_iterations=pair_k_iterations,
                tmap_iterations=tmap_iterations,
                multi_turn_n_turns=multi_turn_n_turns,
            )
        except Exception as e:
            log.error("    %s failed with error: %s", attack_type.value, e)
            from redteam.models.attacks import AttackResult
            result = AttackResult(
                attack_type=attack_type,
                security_score=1.0,
                utility_score=1.0,
                policy_violated=False,
                security_evidence=[f"ERROR: {type(e).__name__}: {e}"],
                utility_evidence=[],
            )

        elapsed = time.time() - t0
        log.info("    %s: violated=%s, security=%.0f%%, utility=%.0f%% (%.1fs)",
                 attack_type.value, result.policy_violated,
                 result.security_score * 100, result.utility_score * 100, elapsed)

        # Semantic judge: add LLM reasoning + confidence to the result
        result = judge_attack(
            result, goal,
            policies=policies or [],
            attacker_base_url=attacker_base_url,
            attacker_model=attacker_model,
        )

        attack_results.append(result)

    return GoalResult(
        goal=goal,
        model_id=model_config.model,
        environment=env,
        dry_run_trace=dry_run_trace,
        attack_results=attack_results,
    )


def _run_attack(
    attack_type: AttackType,
    goal: AttackerGoal,
    env: GeneratedEnvironment,
    dry_run_trace,
    adapter: SyntheticEnvTargetAdapter,
    attacker_base_url: str,
    attacker_model: str,
    pair_n_streams: int,
    pair_k_iterations: int,
    tmap_iterations: int,
    multi_turn_n_turns: int,
) -> AttackResult:
    """Dispatch to the correct attack implementation."""
    common = dict(
        goal=goal,
        env=env,
        dry_run_trace=dry_run_trace,
        adapter=adapter,
        attacker_base_url=attacker_base_url,
        attacker_model=attacker_model,
    )

    if attack_type == AttackType.DIRECT_INJECTION:
        return run_direct_injection(**common)

    elif attack_type == AttackType.PAIR_INJECTION:
        return run_pair(**common, mode="injection",
                        n_streams=pair_n_streams, k_iterations=pair_k_iterations)

    elif attack_type == AttackType.PAIR_ADVERSARIAL:
        return run_pair(**common, mode="adversarial",
                        n_streams=pair_n_streams, k_iterations=pair_k_iterations)

    elif attack_type == AttackType.TMAP:
        return run_tmap(**common, iterations=tmap_iterations)

    elif attack_type == AttackType.MULTI_TURN:
        return run_multi_turn(**common, n_turns=multi_turn_n_turns)

    elif attack_type == AttackType.POISONED_RUNTIME:
        return run_poisoned_runtime(**common)

    elif attack_type == AttackType.MINJA:
        return run_minja(**common)

    raise ValueError(f"Unknown attack type: {attack_type}")


def run_layer1(
    usecase: str,
    policies: list[str],
    risk_cards=None,
    models: list[ModelConfig] | None = None,
    attack_types: list[AttackType] | None = None,
    max_goals: int | None = None,
    attacker_base_url: str | None = None,
    attacker_model: str | None = None,
    pair_n_streams: int = 5,
    pair_k_iterations: int = 3,
    tmap_iterations: int = 10,
    multi_turn_n_turns: int = 4,
):
    """Run the complete Layer 1 pipeline.

    Args:
        usecase: Natural language description of the agent system.
        policies: List of operator policy statements.
        risk_cards: Pre-built RiskCards (None = generate from usecase + policies).
        models: Target models to test (None = use TARGET_MODEL env var).
        attack_types: Attacks to run (None = full suite, all attack types).
        max_goals: Limit number of goals (for quick runs).
        attacker_base_url: Attacker LLM endpoint.
        attacker_model: Attacker model name.

    Returns:
        Layer1Report with all results.
    """
    from redteam.models.report import Layer1Report
    from redteam.models.risk import RiskCard

    attacker_model = attacker_model or os.environ.get("ATTACKER_MODEL") or \
                     os.environ.get("TARGET_MODEL", "qwen3.5:2b")
    attacker_base_url = attacker_base_url or os.environ.get("ATTACKER_BASE_URL") or \
                        os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")

    # Default model
    if not models:
        models = [ModelConfig(
            model=os.environ.get("TARGET_MODEL", "qwen3.5:2b"),
            base_url=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        )]

    # Risk pipeline
    log.info("Risk pipeline: generating risk cards...")
    if not risk_cards:
        risk_cards = generate_risk_cards(usecase, policies, model=attacker_model,
                                         base_url=attacker_base_url)
        log.info("  Generated %d risk cards", len(risk_cards))

    triaged = [triage_risk(c) for c in risk_cards]
    agent_risks = filter_agent_level(triaged)
    log.info("  %d agent-level risks after triage", len(agent_risks))

    all_goals = []
    for risk in agent_risks:
        goals = generate_goals(risk, model=attacker_model, base_url=attacker_base_url)
        # Apply max_goals as a per-risk-card limit so every risk card is tested.
        # Without this, a global limit would skip all goals from later risk cards.
        if max_goals:
            goals = goals[:max_goals]
        all_goals.extend(goals)
    log.info("  Generated %d goals total (max_goals=%s per risk card)",
             len(all_goals), max_goals or "unlimited")

    if not all_goals:
        log.warning("No goals generated — check risk generation and triage")
        return Layer1Report(usecase=usecase, policies=policies, goal_results=[])

    # Per-goal pipeline
    goal_results = []
    total = len(all_goals) * len(models)
    done = 0

    for model_config in models:
        for goal in all_goals:
            done += 1
            log.info("\n[%d/%d] Goal: %s | Model: %s",
                     done, total, goal.attacker_objective[:60], model_config.model)

            result = run_per_goal(
                goal=goal,
                policies=policies,
                model_config=model_config,
                attack_types=attack_types,
                attacker_base_url=attacker_base_url,
                attacker_model=attacker_model,
                pair_n_streams=pair_n_streams,
                pair_k_iterations=pair_k_iterations,
                tmap_iterations=tmap_iterations,
                multi_turn_n_turns=multi_turn_n_turns,
            )
            goal_results.append(result)

    return Layer1Report(
        usecase=usecase,
        policies=policies,
        risk_cards=risk_cards,
        goal_results=goal_results,
    )
