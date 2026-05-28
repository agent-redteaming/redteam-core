"""CLI entry point — `redteam-run` command.

Usage examples:

  # Minimal — generate everything from usecase + policy
  redteam-run \\
    --usecase "Document processing agent that reads and summarizes reports" \\
    --policy "Agents must not send document contents to external addresses"

  # With multiple policies and explicit attacks
  redteam-run \\
    --usecase "Inventory management agent with access to stock and supplier data" \\
    --policy "Agents must not modify stock records without supervisor approval" \\
    --policy "Supplier contact details must never be forwarded externally" \\
    --attacks injection pair_adversarial \\
    --model qwen3.5:2b \\
    --output ./results

  # With attacker model and target model separated
  ATTACKER_BASE_URL=https://my-attacker.example.com/v1 \\
  ATTACKER_MODEL=gemma-2-9b-abliterated \\
  OPENAI_BASE_URL=http://localhost:11434/v1 \\
  TARGET_MODEL=qwen3.5:2b \\
  redteam-run --usecase "..." --policy "..."

  # Quick test: limit to 1 goal, fast attacks only
  redteam-run \\
    --usecase "Data analysis agent" \\
    --policy "Agent must not exfiltrate data to external parties" \\
    --attacks injection \\
    --max-goals 1 \\
    --pair-streams 3

Environment variables:
  OPENAI_BASE_URL       Target model endpoint (default: http://localhost:11434/v1)
  TARGET_MODEL         Target model name (default: qwen3.5:2b)
  ATTACKER_BASE_URL     Attacker model endpoint (defaults to OPENAI_BASE_URL)
  ATTACKER_MODEL        Attacker model name (defaults to TARGET_MODEL)
  TMAP_ITERATIONS       T-MAP generations override
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console

console = Console()

VALID_ATTACKS = [
    "injection",
    "pair_injection",
    "pair_adversarial",
    "tmap",
    "multi_turn",
    "poisoned_runtime",
    "minja",
    "all",
]

ATTACK_NAME_MAP = {
    "injection": "direct_injection",
    "pair_injection": "pair_injection",
    "pair_adversarial": "pair_adversarial",
    "tmap": "tmap",
    "multi_turn": "multi_turn",
    "poisoned_runtime": "poisoned_runtime",
    "minja": "minja",
}


@click.command()
@click.option("--usecase", "-u", required=True,
              help="Natural language description of the agent system.")
@click.option("--policy", "-p", multiple=True,
              help="Operator policy statement (repeat for multiple).")
@click.option("--risk-card", "-r", "risk_card_path", default=None, type=click.Path(exists=True),
              help="Path to YAML file with pre-built RiskCards (optional).")
@click.option("--model", "-m", "models", multiple=True,
              help="Target model to test (repeat for multiple). Default: TARGET_MODEL env var.")
@click.option("--attacks", "-a", multiple=True,
              type=click.Choice(VALID_ATTACKS, case_sensitive=False),
              help=f"Attacks to run (default: derived from goals). Options: {', '.join(VALID_ATTACKS)}")
@click.option("--max-goals", default=None, type=int,
              help="Limit number of goals (for quick testing).")
@click.option("--pair-streams", default=5, type=int, show_default=True,
              help="PAIR: number of parallel streams (paper: 30).")
@click.option("--pair-iterations", default=3, type=int, show_default=True,
              help="PAIR: iterations per stream (paper: 3).")
@click.option("--tmap-iterations", default=10, type=int, show_default=True,
              help="T-MAP: evolutionary generations (paper: 100).")
@click.option("--multi-turn-turns", default=4, type=int, show_default=True,
              help="Multi-turn: number of conversation turns.")
@click.option("--output", "-o", default="./runs", show_default=True,
              help="Directory to save results JSON.")
@click.option("--verbose", "-v", is_flag=True, default=False,
              help="Show DEBUG logs.")
def main(
    usecase: str,
    policy: tuple[str, ...],
    risk_card_path: str | None,
    models: tuple[str, ...],
    attacks: tuple[str, ...],
    max_goals: int | None,
    pair_streams: int,
    pair_iterations: int,
    tmap_iterations: int,
    multi_turn_turns: int,
    output: str,
    verbose: bool,
) -> None:
    """redteam-core Layer 1 — systematic synthetic agent red-teaming."""

    # Logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy http logs unless verbose
    if not verbose:
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

    # Validate policies
    policies = list(policy)
    if not policies:
        console.print("[yellow]Warning: No policies provided. Risk generation may be generic.[/yellow]")

    # Parse attack types
    from redteam.models.attacks import AttackType
    selected_attacks: list[AttackType] | None = None
    if attacks:
        if "all" in attacks:
            selected_attacks = list(AttackType)
        else:
            selected_attacks = [AttackType(ATTACK_NAME_MAP[a.lower()]) for a in attacks]

    # Parse models
    from redteam.models.report import ModelConfig
    target_base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    default_model = os.environ.get("TARGET_MODEL", "qwen3.5:2b")

    model_configs: list[ModelConfig] = []
    for m in models:
        model_configs.append(ModelConfig(model=m, base_url=target_base_url))
    if not model_configs:
        model_configs = [ModelConfig(model=default_model, base_url=target_base_url)]

    # Load RiskCards if provided
    risk_cards = None
    if risk_card_path:
        import yaml
        from redteam.models.risk import (
            RiskCard, RiskSource, RiskConsequence, RiskImpact, RiskControl
        )
        with open(risk_card_path) as f:
            data = yaml.safe_load(f)
        # Support both single card and list
        if isinstance(data, dict) and "risk_card" in data:
            data = [data["risk_card"]]
        elif isinstance(data, dict):
            data = [data]
        risk_cards = []
        for item in data:
            risk_cards.append(RiskCard(
                id=item.get("id", "RC-001"),
                risk_source=RiskSource(**item["risk_source"]),
                risk_consequence=RiskConsequence(**item["risk_consequence"]),
                risk_impact=RiskImpact(**item["risk_impact"]),
                risk_controls=[RiskControl(**c) for c in item.get("risk_controls", [])],
                materialization_conditions=item.get("materialization_conditions", ""),
                policy_references=item.get("policy_references", []),
                framework_references=item.get("framework_references", []),
            ))
        console.print(f"[dim]Loaded {len(risk_cards)} risk card(s) from {risk_card_path}[/dim]")

    # Print banner
    from redteam.report import print_banner, print_goal_result, print_summary, save_report
    print_banner(usecase, policies, model_configs, selected_attacks or [])

    # Run
    from redteam.orchestrator import run_layer1

    attacker_base_url = os.environ.get("ATTACKER_BASE_URL")
    attacker_model = os.environ.get("ATTACKER_MODEL")

    try:
        report = run_layer1(
            usecase=usecase,
            policies=policies,
            risk_cards=risk_cards,
            models=model_configs,
            attack_types=selected_attacks,
            max_goals=max_goals,
            attacker_base_url=attacker_base_url,
            attacker_model=attacker_model,
            pair_n_streams=pair_streams,
            pair_k_iterations=pair_iterations,
            tmap_iterations=tmap_iterations,
            multi_turn_n_turns=multi_turn_turns,
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print summary and save
    print_summary(report)
    save_report(report, Path(output))


if __name__ == "__main__":
    main()
