"""Shared utilities used across the redteam package."""

from __future__ import annotations

import ast
import json
import os
import re

from openai import OpenAI


def get_attacker_client(base_url: str | None = None) -> OpenAI:
    """Return an OpenAI-compatible client pointed at the attacker model endpoint."""
    return OpenAI(
        base_url=base_url or os.environ.get("ATTACKER_BASE_URL")
               or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
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
