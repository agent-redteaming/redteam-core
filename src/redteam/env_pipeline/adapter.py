"""SyntheticEnvTargetAdapter — bridge between attack implementations and the executor.

Attack implementations (PAIR, T-MAP, direct injection, etc.) all need to:
  - Send a prompt to the agent and get back a trace
  - Optionally inject payloads into tool responses
  - Reset environment state between attempts
  - Apply environment refinements (create-then-refine)

This adapter wraps the executor so all attack types work with a standard
interface without knowing they're running against exec'd Python functions.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

from redteam.env_pipeline import executor as _executor
from redteam.env_pipeline.executor import _safe_instantiate
from redteam.evaluation.deterministic import DEFAULT_SUSPICIOUS_TOOLS
from redteam.models.attacks import InjectionScenario
from redteam.models.environment import (
    DryRunTrace,
    EnvironmentRefinement,
    GeneratedEnvironment,
    ToolCall,
)
from redteam.runtime.base import AgentRuntime

log = logging.getLogger(__name__)

# Fields that can carry an injection payload, ordered by preference.
# 'name' is last-resort — it works but can confuse record lookup by name.
_INJECTABLE_FIELDS = [
    "notes", "description", "memo", "comment", "body", "text",
    "details", "status", "routing", "remarks", "instructions", "summary",
    "name",
]


def _inject_into_result(result: Any, payload: str, preferred_fields: list[str]) -> Any:
    """Embed an adversarial payload into a tool return value.

    Operates on dicts, lists of dicts, and Pydantic model instances.
    Picks the first injectable text field in order: preferred_fields → _INJECTABLE_FIELDS.
    If no text field exists, adds a 'notes' key so the payload is always surfaced.
    """
    if isinstance(result, list):
        return [_inject_into_record(r, payload, preferred_fields) for r in result]
    if isinstance(result, dict):
        return _inject_into_record(result, payload, preferred_fields)
    if hasattr(result, "model_dump"):
        return _inject_into_record(result.model_dump(), payload, preferred_fields)
    return result


def _inject_into_record(record: Any, payload: str, preferred_fields: list[str]) -> Any:
    """Embed payload into a single record dict, choosing the best text field."""
    if not isinstance(record, dict):
        return record
    record = dict(record)  # copy — do not mutate original

    # Check preferred fields first (from injection key), then fallback order
    for field in (preferred_fields + _INJECTABLE_FIELDS):
        val = record.get(field)
        if isinstance(val, str):
            record[field] = val + "\n\n" + payload
            return record

    # No text field found — add 'notes' as a new key
    record["notes"] = payload
    return record


class SyntheticEnvTargetAdapter:
    """Wraps the synthetic environment executor for use by attack implementations.

    Lifecycle per evaluation:
      1. __init__: exec() code, load clean state, extract tool schemas
      2. execute(): reset → optionally inject → run agent loop → return trace
      3. apply_refinement(): add tools/data without regenerating (create-then-refine)
      4. snapshot_state(): capture current env state for diffing

    State management:
      - _clean_state: immutable baseline (deep copy of seed data)
      - env_instance: mutable live environment (reset between attempts)
      - exec_globals: the Python namespace with all classes + tool functions
    """

    def __init__(
        self,
        env: GeneratedEnvironment,
        runtime: AgentRuntime,
        model: str,
    ) -> None:
        self._env = env
        self._runtime = runtime
        self._model = model

        # Load and exec the generated code
        self.env_instance, self.exec_globals = _executor._load_environment(env)
        self._tool_schemas = _executor._extract_tool_schemas(self.exec_globals)

        # Store the clean state as JSON (immutable baseline)
        self._clean_state_json: str = json.dumps(
            self.env_instance.model_dump(), default=str
        )

        log.info(
            "Adapter ready for goal %s: %d tools",
            env.goal_id,
            len(self._tool_schemas),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def execute(
        self,
        prompt: str,
        injections: dict[str, str] | None = None,
    ) -> DryRunTrace:
        """Run one agent attempt. Resets env state first.

        Args:
            prompt: The user message to send to the agent.
            injections: Optional dict of {record_id:field → payload_text}.
                        Applied to the environment before the agent runs.
                        Used by injection attacks; None for clean/adversarial prompt runs.
        """
        # Reset environment to clean state
        self._reset()

        # Apply injections if provided
        if injections:
            self._apply_injections(injections)

        return _executor._run_agent_loop(
            self.env_instance,
            self.exec_globals,
            prompt,
            self._tool_schemas,
            self._runtime,
            self._model,
        )

    def apply_refinement(self, refinement: EnvironmentRefinement) -> None:
        """Extend environment with additional tools/data (create-then-refine).

        Called by attack types that need capabilities not in the base env
        (e.g. MINJA needs memory tools, PAIR adversarial may need exfil channel).
        Additive — never replaces existing tools or data.
        """
        if refinement.additional_env_code:
            exec(_executor._strip_code_fences(refinement.additional_env_code), self.exec_globals)
            self._tool_schemas = _executor._extract_tool_schemas(self.exec_globals)
            log.info("Refinement applied for %s: %d tools now available",
                     refinement.attack_type, len(self._tool_schemas))

        if refinement.additional_seed_data_json:
            extra = json.loads(refinement.additional_seed_data_json)
            env_class = type(self.env_instance)
            current = self.env_instance.model_dump()
            for key, value in extra.items():
                if key in current and isinstance(current[key], list) and isinstance(value, list):
                    current[key].extend(value)
                else:
                    current[key] = value
            self.env_instance = _safe_instantiate(env_class, current, self.exec_globals)
            # Update clean state to include the new data
            self._clean_state_json = json.dumps(current, default=str)

    def snapshot_state(self) -> dict:
        """Return current environment state as a dict (for state diffing)."""
        return self.env_instance.model_dump()

    def clean_state(self) -> dict:
        """Return the original clean environment state."""
        return json.loads(self._clean_state_json)

    @property
    def tool_schemas(self) -> list[dict]:
        return self._tool_schemas

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Reset environment to clean state."""
        env_class = type(self.env_instance)
        clean = json.loads(self._clean_state_json)
        self.env_instance = _safe_instantiate(env_class, clean, self.exec_globals)

    def _apply_injections(self, injections: dict[str, str]) -> None:
        """Inject adversarial payloads by wrapping tool functions in exec_globals.

        Previous approach (env state mutation) failed because:
        - Payloads landed in fields (e.g. 'notes') that no existing tool returns
        - The agent never read the injected data during its natural task flow

        This approach wraps every data-reading tool so the payload appears in
        whatever the agent reads — identical to how the classic demo embeds
        the payload in emp002.name, which list_employees() surfaces naturally.
        """
        if not injections:
            return

        # Collect all payloads (typically one, sometimes multiple for escalation)
        payloads = list(injections.values())
        primary_payload = payloads[0] if payloads else ""

        # Prefer the target field from the injection key if it exists in responses
        target_fields = []
        for key_str in injections:
            parts = key_str.split(":", 1)
            if len(parts) == 2:
                target_fields.append(parts[1])

        # Wrap all data-reading tools (skip exfil channels and our refinement tools)
        skip = DEFAULT_SUSPICIOUS_TOOLS | {"get_records_with_notes"}
        wrapped_count = 0
        for name in list(self.exec_globals.keys()):
            obj = self.exec_globals[name]
            if not callable(obj) or isinstance(obj, type) or name.startswith("_"):
                continue
            if name in skip:
                continue

            original = obj

            def _make_wrapper(fn, payload, preferred_fields):
                import functools, inspect

                @functools.wraps(fn)
                def wrapper(env, **kwargs):
                    # Reconstruct positional args from kwargs using original signature.
                    # _call_tool calls us as func(env_instance, **coerced); we forward
                    # to the original which also expects (env, **params).
                    result = fn(env, **kwargs)
                    return _inject_into_result(result, payload, preferred_fields)

                # Override signature so _extract_tool_schemas sees original params.
                try:
                    wrapper.__signature__ = inspect.signature(fn)
                except Exception:
                    pass
                return wrapper

            self.exec_globals[name] = _make_wrapper(original, primary_payload, target_fields)
            wrapped_count += 1

        # Do NOT rebuild tool schemas — the wrapped functions have the same
        # parameters as the originals and functools.wraps + __signature__ ensures
        # schema extraction would see the original signature anyway.
        log.info("Injection wrapped %d tools — payload will appear in their responses", wrapped_count)
