"""HTML report generation for Layer 1 red-team results.

Produces a single self-contained interactive HTML file from a Layer1Report.
No external dependencies — all CSS and JS are inlined.

Usage:
    from redteam.report_html import generate_html_report
    path = generate_html_report(report, "./runs/report.html")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from redteam.models.report import Layer1Report


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_html_report(report: Layer1Report, output_path: str) -> str:
    """Generate a self-contained HTML report from a Layer1Report.

    Args:
        report: The completed Layer1Report from run_layer1().
        output_path: Destination file path (e.g. './runs/report.html').

    Returns:
        The absolute path to the saved HTML file.
    """
    data = _build_report_data(report)
    html = _render_html(data)

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return str(path.resolve())


# ---------------------------------------------------------------------------
# Data serialization
# ---------------------------------------------------------------------------


def _build_report_data(report: Layer1Report) -> dict:
    """Serialize a Layer1Report to the REPORT dict consumed by the HTML JS."""
    from datetime import datetime

    goals = []
    for gr in report.goal_results:
        env = gr.environment
        dry = gr.dry_run_trace

        # Parse seed_data to Python (could be str or already dict/list)
        seed_data = _parse_seed_data(env.seed_data_json)

        attacks_out = []
        for ar in gr.attack_results:
            attacks_out.append({
                "type": ar.attack_type.value,
                "violated": ar.policy_violated,
                "security_score": ar.security_score,
                "utility_score": ar.utility_score,
                "iterations": ar.iterations_used,
                "judge_reasoning": ar.judge_reasoning,
                "judge_confidence": ar.judge_confidence,
                "injection_payloads": ar.injection_payloads,
                "security_checks": _parse_evidence(ar.security_evidence),
                "utility_checks": _parse_evidence(ar.utility_evidence),
                "clean_trace": _serialize_trace(ar.clean_trace),
                "attack_trace": _serialize_trace(ar.attack_trace),
                "state_diff": ar.state_diff,
            })

        goals.append({
            "id": gr.goal.id,
            "objective": gr.goal.attacker_objective,
            "success_criteria": gr.goal.success_criteria,
            "asi_category": gr.goal.asi_category.value,
            "severity": gr.goal.severity,
            "any_violation": gr.any_violation,
            "risk_card_id": gr.goal.risk_card_id,
            "model_id": gr.model_id,
            "environment": {
                "user_task": env.user_task_prompt,
                "model_code": env.pydantic_model_code,
                "tool_code": env.tool_function_code,
                "seed_data": seed_data,
                "expected_tools": env.expected_tool_sequence,
            },
            "dry_run": {
                "tool_calls": [
                    {
                        "name": tc.name,
                        "arguments": tc.arguments,
                        "response": tc.response,
                        "turn_number": tc.turn_number,
                    }
                    for tc in dry.tool_calls
                ],
                "conversation": dry.conversation,
                "final_state": _safe_json_parse(dry.final_state_snapshot),
                "accessed_records": dry.accessed_records,
            },
            "attacks": attacks_out,
        })

    risk_cards_out = []
    for rc in report.risk_cards:
        risk_cards_out.append({
            "id": rc.id,
            "risk_source": {
                "description": rc.risk_source.description,
                "likelihood": rc.risk_source.likelihood,
            },
            "risk_consequence": {
                "description": rc.risk_consequence.description,
                "severity": rc.risk_consequence.severity,
            },
            "risk_impact": {
                "description": rc.risk_impact.description,
                "affected_stakeholders": rc.risk_impact.affected_stakeholders,
                "harm_type": rc.risk_impact.harm_type,
            },
            "risk_controls": [
                {"type": ctrl.type, "description": ctrl.description}
                for ctrl in rc.risk_controls
            ],
            "policy_references": rc.policy_references,
            "framework_references": rc.framework_references,
            "materialization_conditions": rc.materialization_conditions,
        })

    total = len(report.goal_results)
    violations = sum(1 for gr in report.goal_results if gr.any_violation)

    return {
        "meta": {
            "usecase": report.usecase,
            "policies": report.policies,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        },
        "summary": {
            "goals_tested": total,
            "violations": violations,
            "owasp_coverage": report.owasp_coverage,
            "policy_violations": report.policy_violations,
            "model_comparison": report.model_comparison,
        },
        "risk_cards": risk_cards_out,
        "goals": goals,
    }


def _parse_evidence(evidence: list[str]) -> list[dict]:
    """Convert PASS:/FAIL:/SKIP:/ERROR: prefixed strings to structured dicts."""
    result = []
    for item in evidence:
        s = item.strip()
        if s.upper().startswith("PASS:"):
            result.append({"status": "pass", "message": s[5:].strip()})
        elif s.upper().startswith("FAIL:"):
            result.append({"status": "fail", "message": s[5:].strip()})
        elif s.upper().startswith("SKIP:"):
            result.append({"status": "skip", "message": s[5:].strip()})
        elif s.upper().startswith("ERROR:"):
            result.append({"status": "error", "message": s[6:].strip()})
        else:
            # Heuristic: if the string contains "pass" or "✓" treat as pass
            lower = s.lower()
            if "passed" in lower or "✓" in s or "resisted" in lower:
                result.append({"status": "pass", "message": s})
            elif "failed" in lower or "violated" in lower or "✗" in s or "detected" in lower:
                result.append({"status": "fail", "message": s})
            else:
                result.append({"status": "neutral", "message": s})
    return result


def _serialize_trace(trace: Any) -> dict | None:
    """Serialize a DryRunTrace (or None) to a plain dict."""
    if trace is None:
        return None
    return {
        "tool_calls": [
            {
                "name": tc.name,
                "arguments": tc.arguments,
                "response": (tc.response.model_dump() if hasattr(tc.response, "model_dump")
                             else [i.model_dump() if hasattr(i, "model_dump") else i for i in tc.response]
                             if isinstance(tc.response, list) else tc.response),
                "turn_number": tc.turn_number,
            }
            for tc in trace.tool_calls
        ],
        "conversation": trace.conversation,
        "final_state": _safe_json_parse(trace.final_state_snapshot),
        "accessed_records": trace.accessed_records,
    }


def _safe_json_parse(s: str) -> Any:
    """Try to parse a JSON string; return empty dict on failure."""
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


def _parse_seed_data(seed_json: str) -> Any:
    """Parse seed_data_json; handle both dict-of-lists and list forms."""
    if not seed_json:
        return {}
    try:
        return json.loads(seed_json)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _render_html(data: dict) -> str:
    """Return the complete self-contained HTML string."""
    report_json = json.dumps(data, indent=2, ensure_ascii=False, default=str)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Layer 1 Red-Team Report</title>
{_css()}
</head>
<body>

<div class="app-layout">

  <!-- ===== LEFT SIDEBAR ===== -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title">Goals</div>
      <div class="sidebar-sub" id="sidebar-sub"></div>
    </div>
    <div class="sidebar-list" id="sidebar-list"></div>
  </aside>

  <!-- ===== MAIN PANEL ===== -->
  <div class="main-panel">

    <!-- Fixed top header -->
    <div class="header">
      <div>
        <h1><span>Layer 1</span> Red-Team Report</h1>
        <div class="header-meta" id="header-meta"></div>
      </div>
      <div class="header-badges" id="header-badges"></div>
    </div>

    <!-- Pipeline strip -->
    <div class="pipeline-strip" id="pipeline-strip"></div>

    <!-- Content area -->
    <div class="content" id="content-area"></div>

  </div>
</div>

<script>
const REPORT = {report_json};

// ============================================================
// State
// ============================================================
let selectedGoalIdx = null;
let selectedStep = 'overview';

// ============================================================
// Init
// ============================================================
document.addEventListener('DOMContentLoaded', () => {{
  renderHeaderMeta();
  renderSidebar();
  selectOverview();
}});

// ============================================================
// Header
// ============================================================
function renderHeaderMeta() {{
  const meta = REPORT.meta;
  document.getElementById('header-meta').textContent = meta.usecase.slice(0, 80) + (meta.usecase.length > 80 ? '…' : '');

  const s = REPORT.summary;
  const rate = s.goals_tested > 0 ? s.violations / s.goals_tested : 0;
  const badgesEl = document.getElementById('header-badges');
  badgesEl.innerHTML = `
    <span class="badge ${{s.violations > 0 ? 'badge-fail' : 'badge-pass'}}">
      ${{s.violations}}/${{s.goals_tested}} Violated (${{pct(rate)}})
    </span>`;
}}

// ============================================================
// Sidebar
// ============================================================
function renderSidebar() {{
  const sub = document.getElementById('sidebar-sub');
  const s = REPORT.summary;
  sub.textContent = `${{s.goals_tested}} goal${{s.goals_tested !== 1 ? 's' : ''}}`;

  const list = document.getElementById('sidebar-list');
  list.innerHTML = '';

  // Overview item
  const ovItem = el('div', 'sidebar-item sidebar-item-overview active', `
    <div class="sidebar-icon">&#9670;</div>
    <div class="sidebar-item-text">
      <div class="sidebar-item-title">Overview</div>
    </div>
  `);
  ovItem.dataset.idx = 'overview';
  ovItem.addEventListener('click', () => selectOverview());
  list.appendChild(ovItem);

  // Goal items
  REPORT.goals.forEach((goal, i) => {{
    const short = goal.objective.length > 48 ? goal.objective.slice(0, 48) + '…' : goal.objective;
    const item = el('div', 'sidebar-item' + (goal.any_violation ? ' sidebar-item-violated' : ''), `
      <div class="sidebar-item-text">
        <div class="sidebar-item-id">${{goal.id}} <span class="tag ${{severityTag(goal.severity)}}">${{goal.severity}}</span></div>
        <div class="sidebar-item-title">${{esc(short)}}</div>
        <div class="sidebar-item-badges">
          ${{goal.any_violation
            ? '<span class="badge badge-fail" style="font-size:10px;padding:2px 6px">VIOLATED</span>'
            : '<span class="badge badge-pass" style="font-size:10px;padding:2px 6px">passed</span>'
          }}
          <span class="tag tag-asi">${{goal.asi_category}}</span>
        </div>
      </div>
    `);
    item.dataset.idx = String(i);
    item.addEventListener('click', () => selectGoal(i));
    list.appendChild(item);
  }});
}}

// ============================================================
// Navigation
// ============================================================
function selectOverview() {{
  selectedGoalIdx = null;
  selectedStep = 'overview';
  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
  document.querySelector('.sidebar-item-overview').classList.add('active');
  renderPipeline([]);
  renderContent();
}}

function selectGoal(idx) {{
  selectedGoalIdx = idx;
  selectedStep = 'attacks';
  document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.sidebar-item')[idx + 1].classList.add('active');
  renderPipeline(buildSteps(REPORT.goals[idx]));
  renderContent();
}}

function selectStep(stepId) {{
  selectedStep = stepId;
  document.querySelectorAll('.pipeline-step').forEach(s => s.classList.remove('active'));
  const active = document.querySelector(`.pipeline-step[data-step="${{stepId}}"]`);
  if (active) active.classList.add('active');
  renderContent();
}}

// ============================================================
// Pipeline strip
// ============================================================
function buildSteps(goal) {{
  const anyViolated = (goal.attacks || []).some(a => a.violated);
  const attackBadge = anyViolated ? ' 💀' : ' ✓';
  return [
    {{ id: 'risk',    label: 'Risk Card',   num: '1' }},
    {{ id: 'env',     label: 'Environment', num: '2' }},
    {{ id: 'dryrun',  label: 'Dry Run',     num: '3' }},
    {{ id: 'attacks', label: 'Attacks' + attackBadge, num: '4' }},
    {{ id: 'summary', label: 'Summary',     num: '★' }},
  ];
}}

function renderPipeline(steps) {{
  const strip = document.getElementById('pipeline-strip');
  if (steps.length === 0) {{
    strip.innerHTML = `
      <div class="pipeline-step active" data-step="overview" onclick="selectOverview()">
        <div class="step-num" style="background:var(--text)">&#9670;</div>
        <div class="step-label">Overview</div>
      </div>`;
    return;
  }}

  let html = '';
  steps.forEach((step, i) => {{
    const isActive = step.id === selectedStep;
    html += `<div class="pipeline-step${{isActive ? ' active' : ''}}" data-step="${{step.id}}" onclick="selectStep('${{step.id}}')">
      <div class="step-num">${{step.num}}</div>
      <div class="step-label">${{esc(step.label)}}</div>
    </div>`;
    if (i < steps.length - 1) html += `<div class="pipeline-arrow">&rarr;</div>`;
  }});
  strip.innerHTML = html;
}}

// ============================================================
// Content rendering dispatcher
// ============================================================
function renderContent() {{
  const area = document.getElementById('content-area');
  if (selectedGoalIdx === null) {{
    area.innerHTML = renderOverview();
    return;
  }}
  const goal = REPORT.goals[selectedGoalIdx];
  if (selectedStep === 'risk') {{
    area.innerHTML = renderRiskCard(goal);
  }} else if (selectedStep === 'env') {{
    area.innerHTML = renderEnvironment(goal);
  }} else if (selectedStep === 'dryrun') {{
    area.innerHTML = renderDryRun(goal);
  }} else if (selectedStep === 'attacks') {{
    area.innerHTML = renderAllAttacks(goal);
  }} else if (selectedStep === 'summary') {{
    area.innerHTML = renderSummary();
  }}
}}

// ============================================================
// Overview
// ============================================================
function renderOverview() {{
  const s = REPORT.summary;
  const meta = REPORT.meta;
  const rate = s.goals_tested > 0 ? s.violations / s.goals_tested : 0;

  // OWASP bar chart
  const owaspEntries = Object.entries(s.owasp_coverage || {{}});
  let owaspBars = '';
  if (owaspEntries.length > 0) {{
    owaspBars = owaspEntries.sort((a, b) => a[0].localeCompare(b[0])).map(([cat, data]) => {{
      const pctVal = Math.round(data.violation_rate * 100);
      const color = data.violation_rate >= 0.5 ? 'var(--red)' : data.violation_rate > 0 ? 'var(--orange)' : 'var(--green)';
      return `<div class="owasp-row">
        <div class="owasp-label">${{cat}}</div>
        <div class="owasp-bar-wrap">
          <div class="owasp-bar" style="width:${{pctVal}}%;background:${{color}}"></div>
        </div>
        <div class="owasp-stat" style="color:${{color}}">${{data.violations}}/${{data.goals_tested}} (${{pctVal}}%)</div>
      </div>`;
    }}).join('');
  }}

  // Policies list
  const policiesHtml = (meta.policies || []).map(p => `<li>${{esc(p)}}</li>`).join('');

  // Goal cards grid
  const goalCards = REPORT.goals.map((goal, i) => {{
    const violBadge = goal.any_violation
      ? '<span class="badge badge-fail" style="font-size:11px">VIOLATED</span>'
      : '<span class="badge badge-pass" style="font-size:11px">passed</span>';
    return `<div class="goal-card" onclick="selectGoal(${{i}})">
      <div class="goal-card-header">
        <span class="tag tag-asi">${{goal.asi_category}}</span>
        <span class="tag ${{severityTag(goal.severity)}}">${{goal.severity}}</span>
        ${{violBadge}}
      </div>
      <div class="goal-card-id">${{esc(goal.id)}}</div>
      <div class="goal-card-obj">${{esc(goal.objective.slice(0, 100))}}</div>
      <div class="goal-card-attacks">${{(goal.attacks || []).map(a => `<span class="tag tag-attack">${{attackLabel(a.type)}}</span>`).join(' ')}}</div>
    </div>`;
  }}).join('');

  return `
    <div style="max-width:900px">
      <h2 style="font-size:26px;font-weight:700;margin-bottom:6px">Layer 1 Red-Team Report</h2>
      <p style="color:var(--text-dim);margin-bottom:28px;font-size:15px">${{esc(meta.usecase)}}</p>

      <div class="grid-3" style="margin-bottom:24px">
        <div class="stat-card">
          <div class="stat-num">${{s.goals_tested}}</div>
          <div class="stat-label">Goals Tested</div>
        </div>
        <div class="stat-card ${{s.violations > 0 ? 'stat-card-fail' : ''}}">
          <div class="stat-num" style="color:${{s.violations > 0 ? 'var(--red)' : 'var(--green)'}}">${{s.violations}}</div>
          <div class="stat-label">Violations</div>
        </div>
        <div class="stat-card">
          <div class="stat-num" style="color:${{rate > 0.5 ? 'var(--red)' : rate > 0 ? 'var(--orange)' : 'var(--green)'}}">${{pct(rate)}}</div>
          <div class="stat-label">Violation Rate</div>
        </div>
      </div>

      ${{meta.policies && meta.policies.length > 0 ? `
      <div class="card" style="margin-bottom:20px">
        <div class="card-header"><h3>Policies Under Test</h3></div>
        <div class="card-body"><ul style="padding-left:18px;line-height:2">${{policiesHtml}}</ul></div>
      </div>` : ''}}

      ${{owaspBars ? `
      <div class="card" style="margin-bottom:20px">
        <div class="card-header"><h3>OWASP ASI Coverage</h3></div>
        <div class="card-body">${{owaspBars}}</div>
      </div>` : ''}}

      <div class="card" style="margin-bottom:20px">
        <div class="card-header"><h3>Goals</h3><span class="text-dim" style="font-size:13px">Click a card to explore</span></div>
        <div class="card-body">
          <div class="goal-cards-grid">${{goalCards}}</div>
        </div>
      </div>
    </div>`;
}}

// ============================================================
// Risk Card
// ============================================================
function renderRiskCard(goal) {{
  const rc = (REPORT.risk_cards || []).find(r => r.id === goal.risk_card_id);
  if (!rc) {{
    return `<div class="card"><div class="card-body"><p class="text-dim">No risk card data available for ${{esc(goal.risk_card_id)}}.</p></div></div>`;
  }}

  const controlsHtml = (rc.risk_controls || []).map(c => `
    <div style="margin-bottom:8px">
      <span class="tag tag-ctrl-${{c.type}}">${{c.type}}</span>
      <span style="font-size:13px;margin-left:8px">${{esc(c.description)}}</span>
    </div>`).join('');

  const polRefs = (rc.policy_references || []).map(p => `<span class="tag" style="background:#f0f2f7;color:var(--text-dim);border:1px solid var(--border);margin-right:4px">${{esc(p)}}</span>`).join('');
  const fwRefs = (rc.framework_references || []).map(p => `<span class="tag" style="background:#f0f2f7;color:var(--text-dim);border:1px solid var(--border);margin-right:4px">${{esc(p)}}</span>`).join('');

  return `
    <h2 style="margin-bottom:20px">Step 1: Risk Card</h2>

    <div class="card">
      <div class="card-header">
        <h3>RiskCard: ${{esc(rc.id)}}</h3>
        <span class="tag ${{severityTag(rc.risk_consequence.severity)}}">${{rc.risk_consequence.severity}}</span>
      </div>
      <div class="card-body">
        <div class="kv-grid">
          <div class="kv-label">Risk Source</div>
          <div>${{esc(rc.risk_source.description)}}</div>
          <div class="kv-label">Likelihood</div>
          <div><span class="tag ${{severityTag(rc.risk_source.likelihood)}}">${{rc.risk_source.likelihood}}</span></div>
          <div class="kv-label">Consequence</div>
          <div>${{esc(rc.risk_consequence.description)}}</div>
          <div class="kv-label">Severity</div>
          <div><span class="tag ${{severityTag(rc.risk_consequence.severity)}}">${{rc.risk_consequence.severity}}</span></div>
          ${{rc.risk_impact ? `
          <div class="kv-label">Impact</div>
          <div>${{esc(rc.risk_impact.description)}}</div>
          <div class="kv-label">Stakeholders</div>
          <div>${{(rc.risk_impact.affected_stakeholders || []).join(', ')}}</div>
          <div class="kv-label">Harm Type</div>
          <div>${{esc(rc.risk_impact.harm_type || '')}}</div>
          ` : ''}}
          <div class="kv-label">Controls</div>
          <div>${{controlsHtml || '<span class="text-dim">None listed</span>'}}</div>
          ${{polRefs ? `<div class="kv-label">Policy Refs</div><div>${{polRefs}}</div>` : ''}}
          ${{fwRefs ? `<div class="kv-label">Framework Refs</div><div>${{fwRefs}}</div>` : ''}}
          ${{rc.materialization_conditions ? `<div class="kv-label">Materialization</div><div>${{esc(rc.materialization_conditions)}}</div>` : ''}}
        </div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Attacker Goal: ${{esc(goal.id)}}</h3></div>
      <div class="card-body">
        <div class="kv-grid">
          <div class="kv-label">Objective</div>
          <div>${{esc(goal.objective)}}</div>
          <div class="kv-label">Success Criteria</div>
          <div>${{esc(goal.success_criteria)}}</div>
          <div class="kv-label">ASI Category</div>
          <div><span class="tag tag-asi">${{goal.asi_category}}</span></div>
          <div class="kv-label">Severity</div>
          <div><span class="tag ${{severityTag(goal.severity)}}">${{goal.severity}}</span></div>
          <div class="kv-label">Model</div>
          <div><code>${{esc(goal.model_id)}}</code></div>
        </div>
      </div>
    </div>`;
}}

// ============================================================
// Environment
// ============================================================
function renderEnvironment(goal) {{
  const env = goal.environment;
  const seedHtml = renderSeedTable(env.seed_data);
  const toolsHtml = (env.expected_tools || []).map(t =>
    `<span class="tag" style="background:#eef1fb;color:var(--accent);border:1px solid #c7d2fe;margin-right:4px">${{esc(t)}}</span>`
  ).join('');

  return `
    <h2 style="margin-bottom:20px">Step 2: Environment</h2>

    <div class="card">
      <div class="card-header"><h3>User Task Prompt</h3></div>
      <div class="card-body">
        <p style="font-size:15px;font-style:italic;color:var(--text)">&ldquo;${{esc(env.user_task)}}&rdquo;</p>
      </div>
    </div>

    <div class="grid-2">
      <div class="card">
        <div class="card-header"><h3>Pydantic Models</h3></div>
        <div class="card-body"><pre>${{esc(env.model_code || '')}}</pre></div>
      </div>
      <div class="card">
        <div class="card-header"><h3>Tool Functions</h3></div>
        <div class="card-body"><pre>${{esc(env.tool_code || '')}}</pre></div>
      </div>
    </div>

    <div class="card">
      <div class="card-header"><h3>Seed Data</h3></div>
      <div class="card-body">${{seedHtml}}</div>
    </div>

    ${{toolsHtml ? `
    <div class="card">
      <div class="card-header"><h3>Expected Tool Sequence</h3></div>
      <div class="card-body">${{toolsHtml}}</div>
    </div>` : ''}}`;
}}

function renderSeedTable(seedData) {{
  if (!seedData) return '<p class="text-dim">No seed data.</p>';

  // dict-of-lists: {{ employees: [...], outbox: [...] }}
  if (typeof seedData === 'object' && !Array.isArray(seedData)) {{
    const keys = Object.keys(seedData);
    if (keys.length === 0) return '<p class="text-dim">Empty seed data.</p>';

    return keys.map(k => {{
      const rows = seedData[k];
      if (!Array.isArray(rows) || rows.length === 0) {{
        return `<div class="text-dim" style="margin-bottom:8px"><strong>${{esc(k)}}</strong>: (empty)</div>`;
      }}
      return `<div style="margin-bottom:20px">
        <div style="font-weight:600;margin-bottom:8px;font-size:13px;text-transform:uppercase;letter-spacing:.5px;color:var(--text-dim)">${{esc(k)}}</div>
        ${{arrayToTable(rows)}}
      </div>`;
    }}).join('');
  }}

  // list form
  if (Array.isArray(seedData)) {{
    if (seedData.length === 0) return '<p class="text-dim">Empty seed data.</p>';
    return arrayToTable(seedData);
  }}

  // fallback: pretty JSON
  return `<pre>${{esc(JSON.stringify(seedData, null, 2))}}</pre>`;
}}

function cellVal(v) {{
  if (v === null || v === undefined) return '';
  if (typeof v === 'object') return JSON.stringify(v);
  return String(v);
}}

function arrayToTable(rows) {{
  if (!rows || rows.length === 0) return '';
  const cols = Object.keys(rows[0]);
  const thead = `<tr>${{cols.map(c => `<th>${{esc(c)}}</th>`).join('')}}</tr>`;
  const tbody = rows.map(row =>
    `<tr>${{cols.map(c => `<td style="max-width:280px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="${{esc(cellVal(row[c]))}}">${{esc(cellVal(row[c]))}}</td>`).join('')}}</tr>`
  ).join('');
  return `<div class="table-scroll"><table><thead>${{thead}}</thead><tbody>${{tbody}}</tbody></table></div>`;
}}

// ============================================================
// Dry Run
// ============================================================
function renderDryRun(goal) {{
  const dr = goal.dry_run;
  const convHtml = renderConversation(dr.conversation);
  const toolsHtml = renderToolCalls(dr.tool_calls);
  const recsHtml = (dr.accessed_records || []).map(r =>
    `<span class="tag" style="background:#f0f2f7;color:var(--text-dim);border:1px solid var(--border);margin-right:4px">${{esc(r)}}</span>`
  ).join('');

  return `
    <h2 style="margin-bottom:20px">Step 3: Dry Run (Clean Baseline)</h2>
    <p style="color:var(--text-dim);margin-bottom:20px">Agent run against the clean environment. No injections. Establishes baseline behavior.</p>

    ${{recsHtml ? `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header"><h3>Accessed Records</h3></div>
      <div class="card-body">${{recsHtml}}</div>
    </div>` : ''}}

    ${{toolsHtml ? `
    <div class="card">
      <div class="card-header"><h3>Tool Call Trace (${{(dr.tool_calls||[]).length}} calls)</h3></div>
      <div class="card-body">${{toolsHtml}}</div>
    </div>` : ''}}

    <div class="card">
      <div class="card-header"><h3>Full Conversation</h3></div>
      <div class="card-body">
        <div class="raw-trace">${{convHtml}}</div>
      </div>
    </div>`;
}}

// ============================================================
// All-attacks page — one tile per attack type
// ============================================================
function renderAllAttacks(goal) {{
  const attacks = goal.attacks || [];
  if (attacks.length === 0) {{
    return `<h2 style="margin-bottom:20px">Attacks</h2><p class="text-dim">No attacks were run for this goal.</p>`;
  }}

  const tilesHtml = attacks.map((atk, i) => renderAttack(goal, atk, i)).join('');

  return `
    <h2 style="margin-bottom:8px">Step 4: Attacks</h2>
    <p style="color:var(--text-dim);margin-bottom:28px">
      ${{attacks.length}} attack${{attacks.length !== 1 ? 's' : ''}} run against this goal.
      ${{attacks.filter(a => a.violated).length}} violation${{attacks.filter(a => a.violated).length !== 1 ? 's' : ''}} found.
    </p>
    ${{tilesHtml}}`;
}}

// ============================================================
// Single attack tile (used both in all-attacks page and standalone)
// ============================================================
function renderAttack(goal, atk, atkIdx) {{
  const stepNum = 4 + atkIdx;
  const violBadge = atk.violated
    ? '<span class="badge badge-fail">VIOLATED</span>'
    : '<span class="badge badge-pass">passed</span>';

  let topHtml = '';
  const type = atk.type;

  if (type === 'direct_injection') {{
    topHtml = renderInjectionPayloads(atk.injection_payloads);
  }} else if (type === 'pair_adversarial' || type === 'pair_injection') {{
    topHtml = renderPairInfo(atk);
  }}

  const convHtml = atk.attack_trace ? renderConversation(atk.attack_trace.conversation) : '<p class="text-dim">No trace available.</p>';
  const evalHtml = renderEvaluation(atk);

  return `
    <div style="border:2px solid ${{atk.violated ? 'var(--red)' : 'var(--border)'}};border-radius:12px;overflow:hidden;margin-bottom:32px;box-shadow:0 2px 8px rgba(0,0,0,.06)">

      <!-- Tile header -->
      <div style="padding:16px 20px;background:${{atk.violated ? '#fef2f2' : 'var(--surface2)'}};display:flex;align-items:center;justify-content:space-between;gap:12px">
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:18px;font-weight:700;color:${{atk.violated ? 'var(--red)' : 'var(--text-dim)'}}">${{atk.violated ? '💀' : '✓'}}</span>
          <div>
            <div style="font-size:15px;font-weight:700;color:var(--text)">${{esc(attackLabel(type))}}</div>
            <div style="font-size:12px;color:var(--text-dim);margin-top:2px">
              Iterations: ${{atk.iterations}} &nbsp;·&nbsp;
              Security: <strong style="color:${{atk.security_score < 0.5 ? 'var(--red)' : 'var(--green)'}}">${{pct(atk.security_score)}}</strong> &nbsp;·&nbsp;
              Utility: <strong style="color:${{atk.utility_score >= 0.5 ? 'var(--green)' : 'var(--orange)'}}">${{pct(atk.utility_score)}}</strong>
            </div>
          </div>
        </div>
        ${{violBadge}}
      </div>

      <!-- Tile body -->
      <div style="padding:20px;background:var(--surface)">
        ${{topHtml}}

        <div class="card" style="margin-bottom:0">
          <div class="card-header" style="cursor:pointer" onclick="this.nextElementSibling.style.display=this.nextElementSibling.style.display==='none'?'block':'none'">
            <h3>Attack Conversation <span style="font-size:12px;color:var(--text-dim)">(click to toggle)</span></h3>
          </div>
          <div class="card-body" style="display:none">
            <div class="raw-trace">${{convHtml}}</div>
          </div>
        </div>
      </div>

      <!-- Evaluation footer -->
      <div style="padding:20px;background:var(--surface2);border-top:1px solid var(--border)">
        ${{evalHtml}}
      </div>
    </div>`;
}}

function renderInjectionPayloads(payloads) {{
  if (!payloads || payloads.length === 0) return '';

  const items = payloads.map((p, i) => {{
    const fragClass = payloads.length === 1 ? 'frag-trigger' :
      (i === 0 ? 'frag-premise' : i === payloads.length - 1 ? 'frag-trigger' : 'frag-reinforce');
    const fragLabel = payloads.length === 1 ? 'Payload' :
      (i === 0 ? 'Fragment 1: Premise' : i === payloads.length - 1 ? `Fragment ${{i+1}}: Trigger` : `Fragment ${{i+1}}: Reinforce`);
    return `<div class="injection-fragment ${{fragClass}}">
      <div class="frag-label">${{esc(fragLabel)}} — record <code>${{esc(p.target_record_id)}}</code>, field <code>${{esc(p.target_field)}}</code></div>
      <div class="frag-meta">
        <span class="tag tag-pattern">${{esc(p.pattern_type || '')}}</span>
        <span class="tag tag-framing" style="margin-left:4px">${{esc(p.framing_style || '')}}</span>
      </div>
      <div class="frag-payload">${{esc(p.payload_text)}}</div>
    </div>`;
  }}).join('');

  return `
    <div class="card">
      <div class="card-header"><h3>Injection Payloads (${{payloads.length}})</h3></div>
      <div class="card-body">${{items}}</div>
    </div>`;
}}

function renderPairInfo(atk) {{
  // Pull user turn from attack conversation
  const conv = atk.attack_trace && atk.attack_trace.conversation ? atk.attack_trace.conversation : [];
  const userTurn = conv.find(m => m.role === 'user');
  const adversarialPrompt = userTurn ? (typeof userTurn.content === 'string' ? userTurn.content : JSON.stringify(userTurn.content)) : '';

  return `
    <div class="card">
      <div class="card-header"><h3>Adversarial Prompt</h3><span class="text-dim" style="font-size:13px">Iterations: ${{atk.iterations}}</span></div>
      <div class="card-body">
        <div class="callout callout-warn">
          <div class="callout-label">PAIR-refined user prompt</div>
          <div style="white-space:pre-wrap;font-size:13px">${{esc(adversarialPrompt)}}</div>
        </div>
      </div>
    </div>`;
}}

// ============================================================
// Evaluation section
// ============================================================
function renderEvaluation(atk) {{
  const secItems = renderEvidenceList(atk.security_checks);
  const utilItems = renderEvidenceList(atk.utility_checks);

  const secPassed = (atk.security_checks || []).filter(c => c.status === 'pass').length;
  const secTotal = (atk.security_checks || []).length;
  const utilPassed = (atk.utility_checks || []).filter(c => c.status === 'pass').length;
  const utilTotal = (atk.utility_checks || []).length;

  const judgeHtml = renderJudge(atk);
  const stateDiffHtml = renderStateDiff(atk.state_diff);
  const outboxDiffHtml = renderOutboxDiff(atk);

  return `
    <div class="grid-2">
      <div class="card ${{atk.violated ? 'card-fail' : 'card-pass'}}">
        <div class="card-header ${{atk.violated ? 'card-header-fail' : 'card-header-pass'}}">
          <h3>Security — ${{atk.violated ? 'FAIL' : 'PASS'}}</h3>
          <span class="badge ${{atk.violated ? 'badge-fail' : 'badge-pass'}}">${{secPassed}}/${{secTotal}} passed</span>
        </div>
        <div class="card-body">${{secItems}}</div>
      </div>
      <div class="card">
        <div class="card-header">
          <h3>Utility — ${{atk.utility_score >= 0.5 ? 'PASS' : 'FAIL'}}</h3>
          <span class="badge ${{atk.utility_score >= 0.5 ? 'badge-pass' : 'badge-fail'}}">${{utilPassed}}/${{utilTotal}} passed</span>
        </div>
        <div class="card-body">${{utilItems}}</div>
      </div>
    </div>

    ${{judgeHtml}}
    ${{outboxDiffHtml}}
    ${{stateDiffHtml}}`;
}}

function renderEvidenceList(checks) {{
  if (!checks || checks.length === 0) return '<p class="text-dim">No checks recorded.</p>';
  return checks.map(c => {{
    let icon, color;
    if (c.status === 'pass') {{ icon = '✓'; color = 'var(--green)'; }}
    else if (c.status === 'fail') {{ icon = '✗'; color = 'var(--red)'; }}
    else if (c.status === 'error') {{ icon = '!'; color = 'var(--orange)'; }}
    else {{ icon = '—'; color = 'var(--text-dim)'; }}
    return `<div class="evidence-item">
      <div class="evidence-icon" style="color:${{color}}">${{icon}}</div>
      <div>${{esc(c.message)}}</div>
    </div>`;
  }}).join('');
}}

function renderJudge(atk) {{
  if (!atk.judge_reasoning && !atk.judge_confidence) return '';
  const confPct = Math.round((atk.judge_confidence || 0) * 100);
  const confColor = confPct >= 70 ? 'var(--green)' : confPct >= 40 ? 'var(--orange)' : 'var(--red)';
  return `
    <div class="card">
      <div class="card-header"><h3>Judge Assessment</h3></div>
      <div class="card-body">
        ${{atk.judge_reasoning ? `<p style="font-size:13px;line-height:1.7;margin-bottom:16px">${{esc(atk.judge_reasoning)}}</p>` : ''}}
        <div style="display:flex;align-items:center;gap:12px">
          <span style="font-size:13px;color:var(--text-dim);font-weight:500;white-space:nowrap">Confidence</span>
          <div class="conf-bar-wrap">
            <div class="conf-bar" style="width:${{confPct}}%;background:${{confColor}}"></div>
          </div>
          <span style="font-size:13px;font-weight:600;color:${{confColor}};white-space:nowrap">${{confPct}}%</span>
        </div>
      </div>
    </div>`;
}}

function renderStateDiff(diff) {{
  if (!diff || Object.keys(diff).length === 0) return '';
  const json = JSON.stringify(diff, null, 2);
  return `
    <div class="card">
      <div class="card-header">
        <h3>State Diff</h3>
        <button class="detail-close" onclick="toggleCollapse(this)" data-target="state-diff-body">Show</button>
      </div>
      <div class="card-body" id="state-diff-body" style="display:none">
        <pre>${{esc(json)}}</pre>
      </div>
    </div>`;
}}

function renderOutboxDiff(atk) {{
  const cleanMsgs = extractOutboxMessages(atk.clean_trace);
  const attackMsgs = extractOutboxMessages(atk.attack_trace);

  if (cleanMsgs.length === 0 && attackMsgs.length === 0) return '';

  const renderMsgList = (msgs, colorClass) => {{
    if (msgs.length === 0) return '<p class="text-dim" style="font-size:13px">No messages sent.</p>';
    return msgs.map(m => `
      <div class="outbox-msg ${{colorClass}}" style="margin-bottom:12px">
        ${{m.to ? `<div style="font-size:12px"><strong>To:</strong> ${{esc(m.to)}}</div>` : ''}}
        ${{m.subject ? `<div style="font-size:12px"><strong>Subject:</strong> ${{esc(m.subject)}}</div>` : ''}}
        ${{m.body ? `<div style="font-size:12px;margin-top:6px;white-space:pre-wrap;color:var(--text-dim)">${{esc(m.body.slice(0, 400))}}${{m.body.length > 400 ? '…' : ''}}</div>` : ''}}
      </div>`).join('');
  }};

  return `
    <div class="card">
      <div class="card-header"><h3>Outbox Diff</h3></div>
      <div class="card-body">
        <div class="diff-row">
          <div class="diff-col">
            <h4 style="color:var(--green)">Clean Run (${{cleanMsgs.length}} sent)</h4>
            <div class="diff-clean">${{renderMsgList(cleanMsgs, 'outbox-clean')}}</div>
          </div>
          <div class="diff-col">
            <h4 style="color:var(--red)">Attack Run (${{attackMsgs.length}} sent)</h4>
            <div class="diff-attack">${{renderMsgList(attackMsgs, 'outbox-attack')}}</div>
          </div>
        </div>
      </div>
    </div>`;
}}

function extractOutboxMessages(trace) {{
  if (!trace) return [];
  const msgs = [];
  for (const tc of (trace.tool_calls || [])) {{
    if (tc.name === 'send_message' || tc.name === 'send_email') {{
      const args = tc.arguments || {{}};
      msgs.push({{
        to: args.to || args.recipient || args.address || '',
        subject: args.subject || '',
        body: args.body || args.content || args.message || '',
      }});
    }}
  }}
  return msgs;
}}

// ============================================================
// Summary
// ============================================================
function renderSummary() {{
  const s = REPORT.summary;

  // OWASP table
  const owaspRows = Object.entries(s.owasp_coverage || {{}})
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([cat, data]) => {{
      const rate = data.violation_rate;
      const color = rate >= 0.5 ? 'var(--red)' : rate > 0 ? 'var(--orange)' : 'var(--green)';
      return `<tr>
        <td><span class="tag tag-asi">${{cat}}</span></td>
        <td>${{data.goals_tested}}</td>
        <td>${{data.violations}}</td>
        <td style="color:${{color}};font-weight:600">${{pct(rate)}}</td>
      </tr>`;
    }}).join('');

  // Policy violations table
  const pvRows = Object.entries(s.policy_violations || {{}}).map(([policy, goalIds]) => `
    <tr>
      <td style="font-size:13px">${{esc(policy)}}</td>
      <td>${{goalIds.map(id => `<span class="tag" style="background:#fef2f2;color:var(--red);border:1px solid #fecaca">${{esc(id)}}</span>`).join(' ')}}</td>
    </tr>`).join('');

  // Model comparison table
  const mcRows = Object.entries(s.model_comparison || {{}}).map(([modelId, data]) => {{
    const rate = data.violation_rate;
    const color = rate >= 0.5 ? 'var(--red)' : rate > 0 ? 'var(--orange)' : 'var(--green)';
    return `<tr>
      <td><code>${{esc(modelId)}}</code></td>
      <td>${{data.goals_tested}}</td>
      <td style="color:${{color}};font-weight:600">${{pct(rate)}}</td>
      <td>${{pct(data.avg_security_score)}}</td>
      <td>${{pct(data.avg_utility_score)}}</td>
    </tr>`;
  }}).join('');

  return `
    <h2 style="margin-bottom:20px">Summary</h2>

    ${{owaspRows ? `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header"><h3>OWASP ASI Coverage</h3></div>
      <div class="card-body" style="padding:0">
        <table>
          <thead><tr><th>Category</th><th>Tested</th><th>Violations</th><th>Rate</th></tr></thead>
          <tbody>${{owaspRows}}</tbody>
        </table>
      </div>
    </div>` : ''}}

    ${{pvRows ? `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header"><h3>Policy Violations</h3></div>
      <div class="card-body" style="padding:0">
        <table>
          <thead><tr><th>Policy</th><th>Violated by Goals</th></tr></thead>
          <tbody>${{pvRows}}</tbody>
        </table>
      </div>
    </div>` : ''}}

    ${{mcRows ? `
    <div class="card" style="margin-bottom:20px">
      <div class="card-header"><h3>Model Comparison</h3></div>
      <div class="card-body" style="padding:0">
        <table>
          <thead><tr><th>Model</th><th>Goals</th><th>Violation Rate</th><th>Avg Security</th><th>Avg Utility</th></tr></thead>
          <tbody>${{mcRows}}</tbody>
        </table>
      </div>
    </div>` : ''}}`;
}}

// ============================================================
// Shared trace renderers
// ============================================================
function renderConversation(conversation) {{
  if (!conversation || conversation.length === 0) return '<p class="text-dim">No conversation data.</p>';
  return conversation.map(msg => {{
    const role = msg.role || 'unknown';
    // Skip repetitive system prompt
    if (role === 'system') return '';
    const roleClass = roleToClass(role);

    let bodyHtml = '';

    // Tool calls embedded in assistant message
    if (msg.tool_calls && msg.tool_calls.length > 0) {{
      bodyHtml += msg.tool_calls.map(tc => {{
        const fn = tc.function || {{}};
        let argsStr = fn.arguments || '';
        try {{ argsStr = JSON.stringify(JSON.parse(argsStr), null, 2); }} catch(e) {{}}
        return `<div class="tool-call-box" style="margin-bottom:8px">
          <div class="tool-call-header">${{esc(fn.name || '?')}}</div>
          <pre style="margin:0;border:none;border-radius:0;font-size:11px;max-height:120px;overflow:auto">${{esc(argsStr)}}</pre>
        </div>`;
      }}).join('');
    }}

    // Reasoning (qwen chain-of-thought)
    if (msg.reasoning) {{
      const r = msg.reasoning.length > 400 ? msg.reasoning.slice(0, 400) + '…' : msg.reasoning;
      bodyHtml += `<details style="margin-bottom:6px"><summary style="font-size:11px;color:var(--text-dim);cursor:pointer">Reasoning (chain-of-thought)</summary>
        <div style="font-size:12px;color:var(--text-dim);padding:8px 0;white-space:pre-wrap">${{esc(r)}}</div>
      </details>`;
    }}

    // Main content
    const rawContent = extractContent(msg.content);
    if (rawContent) {{
      // Tool response: try to pretty-print JSON
      if (role === 'tool') {{
        let pretty = rawContent;
        try {{
          const parsed = JSON.parse(rawContent);
          pretty = JSON.stringify(parsed, null, 2);
        }} catch(e) {{}}
        const truncated = pretty.length > 800 ? pretty.slice(0, 800) + '\\n…(truncated)' : pretty;
        bodyHtml += `<pre style="font-size:11px;max-height:200px;overflow:auto;margin:0">${{esc(truncated)}}</pre>`;
      }} else {{
        bodyHtml += `<div class="raw-content">${{esc(rawContent)}}</div>`;
      }}
    }}

    if (!bodyHtml) return '';
    return `<div class="raw-msg">
      <span class="raw-role ${{roleClass}}">${{role}}</span>
      ${{msg.tool_call_id ? `<span style="color:var(--text-dim);font-size:11px;margin-left:4px">${{esc(msg.tool_call_id)}}</span>` : ''}}
      <div style="margin-top:6px">${{bodyHtml}}</div>
    </div>`;
  }}).filter(Boolean).join('');
}}

function renderToolCalls(toolCalls) {{
  if (!toolCalls || toolCalls.length === 0) return '';
  return toolCalls.map((tc, i) => `
    <div class="trace-turn">
      <div class="turn-num turn-clean">${{i + 1}}</div>
      <div class="turn-content">
        <div class="turn-tool">${{esc(tc.name)}}</div>
        <div class="tool-call-box" style="margin-top:8px">
          <div class="tool-call-header">Args</div>
          <pre style="margin:0;border:none;border-radius:0;font-size:12px">${{esc(JSON.stringify(tc.arguments, null, 2))}}</pre>
          <div class="tool-call-header" style="border-top:1px solid var(--border)">Response</div>
          <pre style="margin:0;border:none;border-radius:0;font-size:12px">${{esc(JSON.stringify(tc.response, null, 2))}}</pre>
        </div>
      </div>
    </div>`).join('');
}}

// ============================================================
// Utilities
// ============================================================
function esc(str) {{
  if (str == null) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}}

function el(tag, className, innerHTML) {{
  const d = document.createElement(tag);
  d.className = className;
  d.innerHTML = innerHTML;
  return d;
}}

function pct(v) {{
  if (v == null) return '—';
  return Math.round(v * 100) + '%';
}}

function roleToClass(role) {{
  if (role === 'system') return 'raw-system';
  if (role === 'user') return 'raw-user';
  if (role === 'assistant') return 'raw-llm';
  if (role === 'tool') return 'raw-tool';
  return 'raw-system';
}}

function extractContent(content) {{
  if (content == null) return '';
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {{
    return content.map(part => {{
      if (typeof part === 'string') return part;
      if (part && part.type === 'text') return part.text || '';
      return JSON.stringify(part);
    }}).join('\\n');
  }}
  return JSON.stringify(content, null, 2);
}}

function attackLabel(type) {{
  const map = {{
    direct_injection: 'Direct Injection',
    pair_injection: 'PAIR Injection',
    pair_adversarial: 'PAIR Adversarial',
    tmap: 'T-MAP',
    multi_turn: 'Multi-Turn',
    poisoned_runtime: 'Poisoned Runtime',
    minja: 'MINJA',
  }};
  return map[type] || type;
}}

function severityTag(sev) {{
  if (sev === 'critical') return 'tag-critical';
  if (sev === 'high') return 'tag-high';
  if (sev === 'medium') return 'tag-medium';
  return 'tag-low';
}}

function toggleCollapse(btn) {{
  const targetId = btn.dataset.target;
  const target = document.getElementById(targetId);
  if (!target) return;
  if (target.style.display === 'none') {{
    target.style.display = '';
    btn.textContent = 'Hide';
  }} else {{
    target.style.display = 'none';
    btn.textContent = 'Show';
  }}
}}
</script>

</body>
</html>"""


def _css() -> str:
    """Return the complete inline CSS <style> block."""
    return """<style>
  :root {
    --bg: #f8f9fc;
    --surface: #ffffff;
    --surface2: #f0f2f7;
    --border: #e2e5ef;
    --text: #1e2030;
    --text-dim: #6b7194;
    --accent: #4361ee;
    --green: #0d9f4f;
    --red: #dc2626;
    --orange: #e36414;
    --yellow: #b45309;
    --purple: #7c3aed;
    --cyan: #0891b2;
    --sidebar-width: 260px;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    height: 100vh;
    overflow: hidden;
  }

  /* ===== Layout ===== */
  .app-layout {
    display: flex;
    height: 100vh;
    overflow: hidden;
  }

  /* ===== Sidebar ===== */
  .sidebar {
    width: var(--sidebar-width);
    flex-shrink: 0;
    background: var(--surface);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .sidebar-header {
    padding: 16px;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
  }
  .sidebar-title {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: var(--text-dim);
    margin-bottom: 2px;
  }
  .sidebar-sub { font-size: 12px; color: var(--text-dim); }
  .sidebar-list {
    overflow-y: auto;
    flex: 1;
  }
  .sidebar-item {
    padding: 10px 14px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background .1s;
  }
  .sidebar-item:hover { background: var(--surface2); }
  .sidebar-item.active { background: #eef1fb; border-left: 3px solid var(--accent); }
  .sidebar-item-overview { display: flex; align-items: center; gap: 10px; }
  .sidebar-icon { font-size: 14px; color: var(--text-dim); }
  .sidebar-item-id { font-size: 11px; color: var(--text-dim); margin-bottom: 2px; }
  .sidebar-item-title { font-size: 13px; font-weight: 500; line-height: 1.4; margin-bottom: 4px; }
  .sidebar-item-badges { display: flex; gap: 4px; flex-wrap: wrap; }
  .sidebar-item-violated .sidebar-item-title { color: var(--red); }

  /* ===== Main panel ===== */
  .main-panel {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* ===== Header ===== */
  .header {
    border-bottom: 1px solid var(--border);
    padding: 16px 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    flex-shrink: 0;
  }
  .header h1 { font-size: 18px; font-weight: 700; }
  .header h1 span { color: var(--accent); }
  .header-meta { font-size: 12px; color: var(--text-dim); margin-top: 2px; }
  .header-badges { display: flex; gap: 8px; }

  /* ===== Badges ===== */
  .badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
  }
  .badge-fail { background: #fef2f2; color: var(--red); border: 1px solid #fecaca; }
  .badge-pass { background: #f0fdf4; color: var(--green); border: 1px solid #bbf7d0; }

  /* ===== Tags ===== */
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .3px;
    white-space: nowrap;
  }
  .tag-critical { background: #fef2f2; color: var(--red); border: 1px solid #fecaca; }
  .tag-high { background: #fff7ed; color: var(--orange); border: 1px solid #fed7aa; }
  .tag-medium { background: #fefce8; color: var(--yellow); border: 1px solid #fde68a; }
  .tag-low { background: #f0f2f7; color: var(--text-dim); border: 1px solid var(--border); }
  .tag-asi { background: #eef1fb; color: var(--accent); border: 1px solid #c7d2fe; }
  .tag-attack { background: #f5f3ff; color: var(--purple); border: 1px solid #ddd6fe; }
  .tag-pattern { background: #ecfeff; color: var(--cyan); border: 1px solid #a5f3fc; }
  .tag-framing { background: #f5f3ff; color: var(--purple); border: 1px solid #ddd6fe; }
  .tag-ctrl-detect { background: #ecfeff; color: var(--cyan); border: 1px solid #a5f3fc; }
  .tag-ctrl-mitigate { background: #fffbeb; color: var(--yellow); border: 1px solid #fde68a; }
  .tag-ctrl-eliminate { background: #f0fdf4; color: var(--green); border: 1px solid #bbf7d0; }
  .tag-ctrl-evaluate { background: #f5f3ff; color: var(--purple); border: 1px solid #ddd6fe; }

  /* ===== Pipeline strip ===== */
  .pipeline-strip {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 16px 24px;
    overflow-x: auto;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    flex-shrink: 0;
  }
  .pipeline-step {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 14px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    cursor: pointer;
    transition: all .15s;
    white-space: nowrap;
    flex-shrink: 0;
  }
  .pipeline-step:hover, .pipeline-step.active {
    border-color: var(--accent);
    background: #eef1fb;
  }
  .pipeline-step.active { box-shadow: 0 0 0 1px var(--accent); }
  .step-num {
    width: 24px; height: 24px;
    border-radius: 50%;
    background: var(--accent);
    color: #fff;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    flex-shrink: 0;
  }
  .step-label { font-size: 13px; font-weight: 500; }
  .pipeline-arrow {
    font-size: 16px;
    color: var(--text-dim);
    padding: 0 6px;
    flex-shrink: 0;
  }

  /* ===== Content ===== */
  .content {
    padding: 28px 32px;
    overflow-y: auto;
    flex: 1;
  }

  /* ===== Cards ===== */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    margin-bottom: 20px;
    overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .card-fail { border-color: #fecaca; }
  .card-pass { border-color: #bbf7d0; }
  .card-header {
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
  }
  .card-header h3 { font-size: 14px; font-weight: 600; }
  .card-header-fail { background: #fef2f2; }
  .card-header-pass { background: #f0fdf4; }
  .card-body { padding: 18px; }

  /* ===== Stats ===== */
  .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,.06);
  }
  .stat-card-fail { border-color: #fecaca; background: #fef2f2; }
  .stat-num { font-size: 36px; font-weight: 700; line-height: 1; margin-bottom: 6px; }
  .stat-label { font-size: 13px; color: var(--text-dim); }

  /* ===== Goal cards grid ===== */
  .goal-cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 16px; }
  .goal-card {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px;
    cursor: pointer;
    transition: all .15s;
    background: var(--surface2);
  }
  .goal-card:hover { border-color: var(--accent); background: #eef1fb; transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,.08); }
  .goal-card-header { display: flex; gap: 4px; flex-wrap: wrap; margin-bottom: 6px; }
  .goal-card-id { font-size: 11px; color: var(--text-dim); margin-bottom: 4px; }
  .goal-card-obj { font-size: 13px; line-height: 1.5; margin-bottom: 8px; }
  .goal-card-attacks { display: flex; gap: 4px; flex-wrap: wrap; }

  /* ===== OWASP bars ===== */
  .owasp-row { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; font-size: 13px; }
  .owasp-label { width: 60px; font-weight: 600; flex-shrink: 0; }
  .owasp-bar-wrap { flex: 1; background: var(--surface2); border-radius: 4px; height: 10px; overflow: hidden; }
  .owasp-bar { height: 100%; border-radius: 4px; transition: width .3s; }
  .owasp-stat { width: 100px; text-align: right; flex-shrink: 0; font-weight: 600; }

  /* ===== Code blocks ===== */
  pre {
    background: #f4f5fa;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 14px;
    overflow-x: auto;
    font-size: 12px;
    line-height: 1.6;
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', monospace;
    color: var(--text);
    white-space: pre-wrap;
    word-break: break-word;
  }

  /* ===== Tables ===== */
  .table-scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th {
    text-align: left;
    padding: 9px 12px;
    background: #f0f2f7;
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .5px;
    font-size: 11px;
  }
  td { padding: 9px 12px; border-top: 1px solid var(--border); vertical-align: top; }

  /* ===== Grid ===== */
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
  @media (max-width: 900px) { .grid-2 { grid-template-columns: 1fr; } }

  /* ===== KV grid ===== */
  .kv-grid {
    display: grid;
    grid-template-columns: 160px 1fr;
    gap: 8px 16px;
    font-size: 13px;
    align-items: start;
  }
  .kv-label { color: var(--text-dim); font-weight: 500; padding-top: 1px; }

  /* ===== Trace timeline ===== */
  .trace-turn {
    display: flex;
    gap: 14px;
    padding: 12px 0;
    border-bottom: 1px solid var(--border);
  }
  .trace-turn:last-child { border-bottom: none; }
  .turn-num {
    width: 30px; height: 30px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700;
    flex-shrink: 0;
  }
  .turn-clean { background: #f0fdf4; color: var(--green); border: 1px solid #bbf7d0; }
  .turn-attack { background: #fef2f2; color: var(--red); border: 1px solid #fecaca; }
  .turn-content { flex: 1; min-width: 0; }
  .turn-tool { font-weight: 600; font-size: 14px; }

  /* ===== Tool call boxes ===== */
  .tool-call-box {
    border: 1px solid var(--border);
    border-radius: 8px;
    overflow: hidden;
    margin-top: 8px;
  }
  .tool-call-box pre { margin: 0; border: none; border-radius: 0; font-size: 11px; }
  .tool-call-header {
    padding: 6px 12px;
    background: #f0f2f7;
    font-size: 12px;
    font-weight: 600;
    font-family: 'SF Mono', 'Fira Code', monospace;
    border-bottom: 1px solid var(--border);
    color: var(--accent);
  }

  /* ===== Raw trace ===== */
  .raw-trace { font-size: 13px; line-height: 1.6; }
  .raw-msg {
    margin-bottom: 14px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 14px;
  }
  .raw-msg:last-child { border-bottom: none; }
  .raw-role {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .5px;
    margin-bottom: 6px;
    font-family: -apple-system, sans-serif;
  }
  .raw-system { background: #f0f2f7; color: var(--text-dim); }
  .raw-user { background: #eef1fb; color: var(--accent); }
  .raw-llm { background: #f0fdf4; color: var(--green); }
  .raw-tool { background: #f5f3ff; color: var(--purple); }
  .raw-content { white-space: pre-wrap; word-break: break-word; color: var(--text); font-family: inherit; }

  /* ===== Injection fragments ===== */
  .injection-fragment {
    background: #fff7ed;
    border-left: 3px solid var(--red);
    padding: 12px 16px;
    margin: 10px 0;
    border-radius: 0 8px 8px 0;
    font-size: 13px;
  }
  .injection-fragment .frag-label {
    font-weight: 700;
    text-transform: uppercase;
    font-size: 11px;
    letter-spacing: .5px;
    margin-bottom: 6px;
  }
  .frag-meta { margin-bottom: 8px; }
  .frag-payload { white-space: pre-wrap; word-break: break-word; font-size: 13px; color: var(--text); }
  .frag-premise { border-left-color: var(--yellow); background: #fffbeb; }
  .frag-premise .frag-label { color: var(--yellow); }
  .frag-reinforce { border-left-color: var(--orange); background: #fff7ed; }
  .frag-reinforce .frag-label { color: var(--orange); }
  .frag-trigger { border-left-color: var(--red); background: #fef2f2; }
  .frag-trigger .frag-label { color: var(--red); }

  /* ===== Evidence ===== */
  .evidence-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 7px 0;
    font-size: 13px;
    border-bottom: 1px solid var(--border);
  }
  .evidence-item:last-child { border-bottom: none; }
  .evidence-icon { font-size: 15px; flex-shrink: 0; margin-top: 1px; font-weight: 700; }

  /* ===== Confidence bar ===== */
  .conf-bar-wrap { flex: 1; background: var(--surface2); border-radius: 4px; height: 10px; overflow: hidden; }
  .conf-bar { height: 100%; border-radius: 4px; transition: width .4s; }

  /* ===== Callout ===== */
  .callout {
    border-radius: 8px;
    padding: 14px;
    font-size: 13px;
    line-height: 1.7;
  }
  .callout-warn { background: #fffbeb; border: 1px solid #fde68a; }
  .callout-label {
    font-weight: 700;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: .5px;
    color: var(--yellow);
    margin-bottom: 8px;
  }

  /* ===== Diff view ===== */
  .diff-row { display: flex; gap: 20px; }
  .diff-col { flex: 1; }
  .diff-col h4 { font-size: 12px; text-transform: uppercase; letter-spacing: .5px; margin-bottom: 8px; }
  .diff-clean { border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px; background: #f0fdf4; font-size: 13px; }
  .diff-attack { border: 1px solid #fecaca; border-radius: 8px; padding: 12px; background: #fef2f2; font-size: 13px; }

  /* ===== Misc ===== */
  .text-dim { color: var(--text-dim); }

  /* ===== Button ===== */
  .detail-close {
    background: none;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    cursor: pointer;
    font-size: 12px;
    color: var(--text-dim);
  }
  .detail-close:hover { background: var(--surface2); }
</style>"""
