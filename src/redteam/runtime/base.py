"""AgentRuntime Protocol — the only abstraction over LLM API differences.

Both Chat Completions and Responses API (with function-type tools) work
the same way conceptually: client sends messages + tool schemas, model
returns either tool_calls or a final text response, client executes tools
and sends results back.

The format differences between APIs are encapsulated here:
  - Chat Completions: messages[], tool_calls in choices[0].message,
    results via {"role": "tool", "tool_call_id": ..., "content": ...}
  - Responses API: input[], function_call items in output[],
    results via {"type": "function_call_output", "call_id": ..., "output": ...}

Everything else in redteam-core (env pipeline, attacks, evaluation) uses
AgentRuntime and is completely API-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel


class Message(TypedDict, total=False):
    """Conversation message — role + content, with optional tool call fields."""
    role: str
    content: str | list | None
    tool_calls: list[dict]          # present when role=assistant + model called tools
    tool_call_id: str               # present when role=tool (Chat Completions)
    name: str                       # present when role=tool (some formats)


class ToolCall(TypedDict):
    """A tool invocation from the model — fully parsed, ready to execute."""
    id: str              # call_id for routing results back
    name: str            # function name
    arguments: dict      # already parsed from JSON — no json.loads needed by callers


class AgentTurn(BaseModel):
    """The result of one model call.

    Either the model wants to call tools (tool_calls is non-empty, final_text is None)
    or it produced a final response (final_text is non-None, tool_calls is empty).
    The agent loop continues while tool_calls is non-empty.
    """

    tool_calls: list[ToolCall] = []
    final_text: str | None = None
    raw_response: Any = None         # provider-specific, for debugging


class AgentRuntime(Protocol):
    """Pluggable LLM client interface.

    Implementations:
      ChatCompletionsRuntime  — default, works with Ollama/vLLM/OpenAI
      ResponsesAPIRuntime     — OpenAI Responses API (stub, future)

    The only things that differ between implementations:
      - complete(): how to send messages + tools, how to parse the response
      - append_tool_result(): how to format the tool result back into messages
    """

    def complete(
        self,
        messages: list[Message],
        tools: list[dict],
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs: Any,
    ) -> AgentTurn:
        """Single model call. Returns tool_calls or final_text, never both."""
        ...

    def append_tool_result(
        self,
        messages: list[Message],
        tool_call: ToolCall,
        result: str,
    ) -> list[Message]:
        """Append a tool result to the message history.

        Handles format differences:
          Chat Completions: {"role": "tool", "tool_call_id": ..., "content": result}
          Responses API:    {"type": "function_call_output", "call_id": ..., "output": result}
        """
        ...
