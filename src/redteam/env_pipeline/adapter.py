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
from redteam.models.attacks import InjectionScenario
from redteam.models.environment import (
    DryRunTrace,
    EnvironmentRefinement,
    GeneratedEnvironment,
    ToolCall,
)
from redteam.runtime.base import AgentRuntime

log = logging.getLogger(__name__)


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
        """Apply injections in {record_id:field → payload_text} format.

        Locates each target record in the live environment and appends
        the payload to the specified field.
        """
        env_data = self.env_instance.model_dump()

        for key_str, payload_text in injections.items():
            parts = key_str.split(":", 1)
            if len(parts) != 2:
                log.warning("Invalid injection key format (expected id:field): %s", key_str)
                continue
            target_id, target_field = parts

            injected = False
            for field_value in env_data.values():
                if not isinstance(field_value, list):
                    continue
                for record in field_value:
                    if isinstance(record, dict) and record.get("id") == target_id:
                        if target_field in record:
                            record[target_field] = f"{record[target_field]}\n\n{payload_text}"
                            log.info("Injected into %s.%s", target_id, target_field)
                            injected = True

            if not injected:
                log.warning("Injection target not found: record=%s field=%s", target_id, target_field)

        # Re-instantiate environment with injected data
        env_class = type(self.env_instance)
        self.env_instance = _safe_instantiate(env_class, env_data, self.exec_globals)
