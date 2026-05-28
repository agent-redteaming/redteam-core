"""ResponsesAPIRuntime — OpenAI Responses API runtime (stub).

Same client-side tool execution semantics as Chat Completions for
function-type tools. Format differences:
  - Input: input[] instead of messages[]
  - Tool calls: output items with type="function_call"
  - Results: {"type": "function_call_output", "call_id": ..., "output": ...}

Wired up when explicitly needed. Not the default for Layer 1
(Chat Completions has better local model support via Ollama/vLLM).

To switch: pass ResponsesAPIRuntime() instead of ChatCompletionsRuntime()
to SyntheticEnvTargetAdapter — no other changes needed.
"""

from __future__ import annotations

import json
import os
from typing import Any

from redteam.runtime.base import AgentTurn, Message, ToolCall


class ResponsesAPIRuntime:
    """OpenAI Responses API runtime.

    Stub implementation — to be completed when Responses API
    compatibility is needed for specific model providers.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            )
        except ImportError as e:
            raise ImportError("openai package required for ResponsesAPIRuntime") from e

    def complete(
        self,
        messages: list[Message],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AgentTurn:
        """Single responses.create() call with function-type tools.

        Uses input[] (not messages[]), returns function_call items in output[].
        The client still executes the functions — same semantics as Chat Completions.
        """
        response = self._client.responses.create(
            model=model,
            input=messages,             # Responses API uses input[], not messages[]
            tools=tools,
            **kwargs,
        )

        tool_calls: list[ToolCall] = []
        for item in response.output:
            if item.type == "function_call":
                try:
                    arguments = json.loads(item.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                tool_calls.append(
                    ToolCall(
                        id=item.call_id,
                        name=item.name,
                        arguments=arguments,
                    )
                )

        final_text = response.output_text if not tool_calls else None
        return AgentTurn(
            tool_calls=tool_calls,
            final_text=final_text,
            raw_response=response,
        )

    def append_tool_result(
        self,
        messages: list[Message],
        tool_call: ToolCall,
        result: str,
    ) -> list[Message]:
        """Append tool result in Responses API format.

        Uses function_call_output type instead of role=tool.
        """
        return list(messages) + [
            {
                "type": "function_call_output",
                "call_id": tool_call["id"],
                "output": result,
            }
        ]
