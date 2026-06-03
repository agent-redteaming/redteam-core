"""Shared utilities used across the redteam package."""

from __future__ import annotations

import ast
import json
import os
import re

import httpx
from openai import OpenAI

# Default timeouts for the attacker endpoint.
# Connect: 60s — server may be busy processing the previous request.
# Read: 600s — LLM inference can take a long time for long outputs.
# Overridable via ATTACKER_CONNECT_TIMEOUT / ATTACKER_READ_TIMEOUT env vars.
_ATTACKER_TIMEOUT = httpx.Timeout(
    connect=float(os.environ.get("ATTACKER_CONNECT_TIMEOUT", "60")),
    read=float(os.environ.get("ATTACKER_READ_TIMEOUT", "600")),
    write=60.0,
    pool=60.0,
)


def generator_temperature(default: float = 0.4) -> float:
    """Temperature for structured generation calls (risk cards, goals, environments).

    Override at runtime: GENERATOR_TEMPERATURE=0.6 redteam-run ...
    or set via --generator-temperature CLI flag (which sets this env var).
    """
    return float(os.environ.get("GENERATOR_TEMPERATURE", str(default)))


def attacker_temperature(default: float = 0.7) -> float:
    """Temperature for attack generation calls (PAIR, injection payloads, directives, etc.).

    Override at runtime: ATTACKER_TEMPERATURE=1.0 redteam-run ...
    or set via --attacker-temperature CLI flag (which sets this env var).
    """
    return float(os.environ.get("ATTACKER_TEMPERATURE", str(default)))


def get_attacker_client(base_url: str | None = None) -> OpenAI:
    """Return an OpenAI-compatible client pointed at the attacker model endpoint.

    Uses a 60s connect timeout (default OpenAI is ~5s) to handle cases where
    the server is busy processing a previous request from another PAIR stream.
    Overridable via ATTACKER_CONNECT_TIMEOUT / ATTACKER_READ_TIMEOUT env vars.
    """
    return OpenAI(
        base_url=base_url or os.environ.get("ATTACKER_BASE_URL")
               or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
        timeout=_ATTACKER_TIMEOUT,
        max_retries=2,  # OpenAI-level retry with backoff before our PAIR-level retry
    )


def fix_json(text: str) -> str:
    """Best-effort repair of common JSON issues in LLM output."""
    text = text.strip()
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    try:
        obj = ast.literal_eval(text)
        return json.dumps(obj)
    except (ValueError, SyntaxError):
        pass
    fixed = re.sub(r",\s*([}\]])", r"\1", text.replace("'", '"'))
    try:
        json.loads(fixed)
        return fixed
    except json.JSONDecodeError:
        return text


def strip_code_fences(code: str) -> str:
    """Strip markdown code fences from LLM output."""
    code = code.strip()
    if code.startswith("```"):
        code = re.sub(r"^```(?:python|json)?\s*", "", code)
        code = re.sub(r"\s*```$", "", code)
    return code.strip()
