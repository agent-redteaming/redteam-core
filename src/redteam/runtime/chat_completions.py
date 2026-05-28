"""ChatCompletionsRuntime — default AgentRuntime using /chat/completions.

Works with any OpenAI-compatible endpoint:
  - Ollama:  OPENAI_BASE_URL=http://localhost:11434/v1
  - vLLM:    OPENAI_BASE_URL=http://localhost:8000/v1
  - OpenAI:  OPENAI_BASE_URL=https://api.openai.com/v1 (default)

Tool execution loop:
  1. Send messages + tool schemas to /chat/completions
  2. Model returns tool_calls (intent only — model never executes tools)
  3. We execute the tool functions (the synthetic Python functions)
  4. Send results back as role=tool messages
  5. Repeat until model produces final text (no tool_calls)

The model is just the decision-maker. We are the runtime.
"""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI

from redteam.runtime.base import AgentTurn, Message, ToolCall


class ChatCompletionsRuntime:
    """OpenAI-compatible Chat Completions runtime.

    Configurable via environment variables (same as agent-policy-redteam's executor):
      OPENAI_BASE_URL — defaults to http://localhost:11434/v1 (Ollama)
      OPENAI_API_KEY  — defaults to "ollama" (dummy for local servers)
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self._client = OpenAI(
            base_url=base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
            api_key=api_key or os.environ.get("OPENAI_API_KEY", "ollama"),
        )

    def complete(
        self,
        messages: list[Message],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AgentTurn:
        """Single /chat/completions call. Returns tool_calls or final_text."""
        kwargs_combined: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs,
        }

        if tools:
            # Wrap in OpenAI function format if not already
            openai_tools = []
            for t in tools:
                if t.get("type") == "function":
                    openai_tools.append(t)
                else:
                    # Plain dict with name/description/parameters
                    openai_tools.append({"type": "function", "function": t})
            kwargs_combined["tools"] = openai_tools

        response = self._client.chat.completions.create(**kwargs_combined)
        msg = response.choices[0].message

        tool_calls: list[ToolCall] = []
        for tc in msg.tool_calls or []:
            try:
                arguments = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                arguments = {}
            tool_calls.append(
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
            )

        return AgentTurn(
            tool_calls=tool_calls,
            final_text=msg.content if not tool_calls else None,
            raw_response=response,
        )

    def append_tool_result(
        self,
        messages: list[Message],
        tool_call: ToolCall,
        result: str,
    ) -> list[Message]:
        """Append tool result in Chat Completions format.

        Requires the assistant message with tool_calls to already be in messages
        (the model's turn that triggered this tool call).
        """
        return list(messages) + [
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": result,
            }
        ]
