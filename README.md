# redteam-core

> ⚠️ **Under active development.** APIs, output formats, and attack implementations are changing frequently. Not yet suitable for production use.

Systematic synthetic red-teaming for AI agents. Given a usecase description and operator policies, `redteam-core` automatically generates a threat model, builds purpose-built synthetic environments, and runs a full suite of attacks — all without touching real tools or data.

---

## What it does

Most agent red-teaming today is manual: someone writes a prompt, runs it against an agent, observes the result. `redteam-core` automates the entire chain:

1. **Threat modelling** — LLM generates risk cards from your usecase + policies
2. **Goal generation** — each risk becomes 2–3 specific, measurable attacker objectives
3. **Synthetic environment** — per goal: Pydantic models, tool functions, seed data purpose-built for that attack
4. **Dry run** — clean baseline trace establishing normal agent behavior
5. **Attack suite** — 6 attack types run against every goal
6. **Evaluation** — deterministic checks + LLM judge + state diff
7. **HTML report** — interactive drill-down: risk cards → goals → attack tabs

All of this is **Layer 1** — fully synthetic, no real infrastructure needed. Designed as the foundation for Layer 2 (real MCP servers + MITM) and Layer 3 (runtime gateway enforcement).

---

## Attack types

| Attack | OWASP ASI | What it tests |
|--------|-----------|---------------|
| `direct_injection` | ASI-04 | Adversarial payload embedded in tool response (one-shot) |
| `pair_injection` | ASI-04 | PAIR iteratively refines an injection payload in tool responses |
| `pair_adversarial` | ASI-01 | PAIR iteratively refines a social-engineering user prompt |
| `multi_turn` | ASI-09 | Progressive escalation across a multi-turn conversation |
| `poisoned_runtime` | ASI-02 | Hidden directive in tool description (`<IMPORTANT>...</IMPORTANT>`) |
| `minja` | ASI-06 | Two-phase memory injection (Phase 1: seed; Phase 2: exploit) |
| `tmap` _(opt-in)_ | ASI-01 | MAP-Elites archive exploring 64 attack styles (expensive) |

---

## Quick start

```bash
# Install
git clone https://github.com/agent-redteaming/redteam-core
cd redteam-core
uv sync

# Requires: Ollama running locally (target model) + an attacker model endpoint
export TARGET_MODEL=qwen3.5:2b
export OPENAI_BASE_URL=http://localhost:11434/v1
export ATTACKER_MODEL=gemma-2-9b-abliterated
export ATTACKER_BASE_URL=https://your-attacker-endpoint/v1

# Run
uv run redteam-run \
  --usecase "A supply chain agent that tracks vendor orders and approves purchase requests." \
  --policy "Agents must not approve purchase orders above \$50,000 without human sign-off" \
  --policy "Vendor contact information must not be shared externally" \
  --max-goals 1 \
  --output ./runs \
  --output-html
```

The HTML report opens at `runs/layer1_report_<timestamp>.html`.

---

## CLI reference

```
redteam-run [OPTIONS]

  -u, --usecase TEXT              Natural language description of the agent system
  -p, --policy TEXT               Operator policy (repeat for multiple)
  -m, --model TEXT                Target model(s) to test
  -a, --attacks [injection|pair_injection|pair_adversarial|tmap|multi_turn|poisoned_runtime|minja|all]
                                  Attacks to run (default: all except tmap)
  --tmap                          Also run T-MAP (expensive: ~150 LLM calls/goal)
  --max-goals INT                 Goals per risk card (default: unlimited)
  --pair-streams INT              PAIR parallel streams (default: 5)
  --pair-iterations INT           PAIR iterations per stream (default: 3)
  --multi-turn-turns INT          Multi-turn conversation turns (default: 4)
  --generator-temperature FLOAT   LLM temperature for risk/goal/env generation (default: 0.4)
  --attacker-temperature FLOAT    LLM temperature for attack content generation (default: 0.7)
  --output TEXT                   Output directory (default: ./runs)
  --output-html                   Also generate interactive HTML report
  -v, --verbose                   Show DEBUG logs
```

Environment variable overrides: `GENERATOR_TEMPERATURE`, `ATTACKER_TEMPERATURE`, `TARGET_MODEL`, `ATTACKER_MODEL`, `ATTACKER_BASE_URL`, `OPENAI_BASE_URL`.

---

## Pipeline

```
Usecase + Policies
       │
       ▼
  Risk Cards  ──── LLM generates 2-4 threat cards per run
       │
       ▼
  Attacker Goals  ── 2-3 per risk card, each with specific success_criteria
       │
       ▼  (per goal × model)
  Synthetic Environment
    ├── Pydantic models (entities, relationships)
    ├── Tool functions (read tools + privileged action tools)
    └── Seed data (values spanning policy thresholds)
       │
       ▼
  Dry Run  ──── clean baseline trace, accessed records
       │
       ▼
  Attack Suite  ──── 6 attacks in parallel tabs
    ├── Payload embedded in tool responses (injection, pair_injection)
    ├── Adversarial user prompts (pair_adversarial, tmap)
    ├── Multi-turn escalation (multi_turn)
    ├── Tool description poisoning (poisoned_runtime)
    └── Memory injection (minja)
       │
       ▼
  Evaluation
    ├── Deterministic: unexpected suspicious tool calls, state diffs,
    │   sensitive field mutations, outbox changes
    └── LLM judge: reasoning + confidence (never overrides deterministic)
       │
       ▼
  HTML Report  ──── interactive: risk cards → goals → attack tabs
```

---

## Report

An interactive single-file HTML report is generated with `--output-html`. It includes:

- **Overview** — usecase, policies, OWASP ASI coverage chart, goal tiles
- **Risk Cards** — full threat model with controls, likelihood, severity
- **Per-goal pipeline** — Environment → Dry Run → Attacks (tabbed by attack type)
  - Each attack tab shows: attack-specific metadata (payloads, prompts, turn sequences), full conversation trace, security/utility evidence, judge reasoning, state diff
- **Summary** — OWASP table (attacks run × violations per ASI category), policy violations, model comparison

A sample report from a supply chain scenario is in [`runs/layer1_report_20260602_193727.html`](runs/layer1_report_20260602_193727.html).

---

## Architecture

```
src/redteam/
├── cli.py                    # redteam-run entry point
├── orchestrator.py           # pipeline coordinator
├── utils.py                  # shared helpers (client, temperature, JSON utils)
├── risk_pipeline/
│   ├── risk_generator.py     # usecase → risk cards
│   ├── goal_generator.py     # risk card → attacker goals
│   └── triage.py             # enforcement level classification
├── env_pipeline/
│   ├── env_generator.py      # goal → synthetic environment (LLM)
│   ├── executor.py           # exec() generated code, run agent loop
│   └── adapter.py            # attack ↔ environment bridge
├── attacks/
│   ├── injection.py          # direct injection (observe-then-inject + tool wrapping)
│   ├── pair.py               # PAIR adversarial + injection (JailbreakingLLMs port)
│   ├── tmap.py               # T-MAP MAP-Elites (faithful port)
│   ├── multi_turn.py         # progressive escalation across turns
│   ├── poisoned_runtime.py   # tool description poisoning
│   └── minja.py              # two-phase memory injection (MINJA port)
├── evaluation/
│   ├── deterministic.py      # 5 security + 5 utility checks
│   └── judge.py              # LLM post-hoc reasoning
├── models/                   # Pydantic models: risk, environment, attacks, report
├── runtime/                  # AgentRuntime abstraction (Chat Completions / Responses API)
└── report_html.py            # self-contained HTML report generator
```

---

## Development

```bash
uv run pytest tests/unit/    # 105 unit tests, no LLM needed
uv run pytest tests/real/    # requires Ollama + attacker endpoint
```

---

## Known limitations

- **Layer 1 only**: synthetic environments, not real MCP servers or production data
- **Small models**: `qwen3.5:2b` resists most attacks; use `qwen2.5:14b` or larger to see more violations
- **Generator quality**: Gemma-2-9B sometimes generates mismatched field names or invalid Python — handled by fallbacks but causes hollow environments
- **T-MAP cost**: 64 seed cells × iterations = 150+ LLM calls per goal; excluded from default suite
