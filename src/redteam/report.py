"""Report generation — console display and JSON persistence.

Produces:
  - Rich console output (live during run, summary at end)
  - JSON file (full results for post-processing)
  - Optional text summary file
"""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from redteam.models.report import GoalResult, Layer1Report

console = Console()


def print_goal_result(result: GoalResult) -> None:
    """Print one goal result to console immediately after it completes."""
    violated = result.any_violation
    model_short = result.model_id.split("/")[-1]

    status = Text("💀 VIOLATED", style="bold red") if violated else Text("🛡️  RESISTED", style="bold green")

    console.print(f"\n  {status}  [{model_short}] {result.goal.attacker_objective[:70]}...")
    console.print(f"          Security: {result.worst_security_score:.0%} | Utility: {result.best_utility_score:.0%} | "
                  f"ASI: {result.goal.asi_category.value}")

    if violated:
        for r in result.attack_results:
            if r.policy_violated:
                console.print(f"          ↳ {r.attack_type.value}: {r.security_evidence[0] if r.security_evidence else ''}")


def print_summary(report: Layer1Report) -> None:
    """Print final summary report to console."""
    console.print()
    console.rule("[bold cyan]Layer 1 Red-Team Report[/bold cyan]")
    console.print()

    # Overview
    total = len(report.goal_results)
    violated = sum(1 for r in report.goal_results if r.any_violation)
    console.print(f"  [dim]Usecase:[/dim]  {report.usecase}")
    console.print(f"  [dim]Goals tested:[/dim]  {total}")
    console.print(f"  [dim]Violations:[/dim]  {violated}/{total} "
                  f"({'[red]' if violated else '[green]'}{violated/total:.0%}[/])")
    console.print()

    # OWASP coverage table
    coverage = report.owasp_coverage
    if coverage:
        table = Table(title="OWASP ASI Coverage", border_style="cyan", show_lines=True)
        table.add_column("Category", style="dim")
        table.add_column("Tested", justify="center")
        table.add_column("Violations", justify="center")
        table.add_column("Rate", justify="center")

        for cat, data in sorted(coverage.items()):
            rate = data["violation_rate"]
            rate_style = "red bold" if rate >= 0.5 else ("yellow" if rate > 0 else "green")
            table.add_row(
                cat,
                str(data["goals_tested"]),
                str(data["violations"]),
                Text(f"{rate:.0%}", style=rate_style),
            )
        console.print(table)
        console.print()

    # Model comparison (if multiple models)
    comparison = report.model_comparison
    if len(comparison) > 1:
        table = Table(title="Model Comparison", border_style="cyan", show_lines=True)
        table.add_column("Model")
        table.add_column("Goals", justify="center")
        table.add_column("Violation Rate", justify="center")
        table.add_column("Avg Security", justify="center")
        table.add_column("Avg Utility", justify="center")

        for model_id, data in sorted(comparison.items()):
            rate = data["violation_rate"]
            rate_style = "red bold" if rate >= 0.5 else ("yellow" if rate > 0 else "green")
            table.add_row(
                model_id.split("/")[-1],
                str(data["goals_tested"]),
                Text(f"{rate:.0%}", style=rate_style),
                f"{data['avg_security_score']:.0%}",
                f"{data['avg_utility_score']:.0%}",
            )
        console.print(table)
        console.print()

    # Per-goal breakdown
    table = Table(title="Per-Goal Results", border_style="cyan", show_lines=True)
    table.add_column("Goal", max_width=50)
    table.add_column("ASI", justify="center")
    table.add_column("Severity", justify="center")
    table.add_column("Violated?", justify="center")
    table.add_column("Attacks", justify="center")

    for r in report.goal_results:
        violated_cell = Text("YES", style="red bold") if r.any_violation else Text("no", style="green")
        sev_style = "red" if r.goal.severity == "critical" else ("yellow" if r.goal.severity == "high" else "dim")
        successful = r.successful_attack_types
        attacks_str = ", ".join(a.replace("_", " ") for a in successful) if successful else "—"

        table.add_row(
            r.goal.attacker_objective[:48],
            r.goal.asi_category.value,
            Text(r.goal.severity, style=sev_style),
            violated_cell,
            attacks_str[:30],
        )

    console.print(table)

    console.print()
    console.print("[dim]NOTE: Results reflect model susceptibility in a controlled synthetic environment.[/dim]")
    console.print("[dim]Layer 1 does not test real tools or real data — that is Layer 2.[/dim]")
    console.print()


def save_report(report: Layer1Report, output_dir: Path) -> Path:
    """Save full report as JSON. Returns the saved path."""
    output_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / f"layer1_report_{timestamp}.json"

    # Serialize to JSON (Pydantic handles it)
    report_data = json.loads(report.model_dump_json())

    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2)

    console.print(f"\n[dim]Full report saved to:[/dim] [cyan]{report_path}[/cyan]")
    return report_path


def print_banner(usecase: str, policies: list[str], models: list, attacks: list) -> None:
    """Print run configuration banner at start."""
    lines = Text()
    lines.append("Usecase    ", style="dim")
    lines.append(f"{usecase[:60]}\n")
    lines.append("Policies   ", style="dim")
    lines.append(f"{len(policies)} policy statement(s)\n")
    lines.append("Models     ", style="dim")
    lines.append(f"{', '.join(m.model for m in models)}\n")
    lines.append("Attacks    ", style="dim")
    lines.append(f"{', '.join(a.value for a in attacks) if attacks else 'derived from goals'}")

    console.print(Panel(
        lines,
        title="[bold cyan]redteam-core Layer 1[/bold cyan]",
        border_style="cyan",
        padding=(1, 2),
    ))
    console.print()
