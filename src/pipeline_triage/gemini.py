"""Gemini behind the TriageModel interface, through LangChain's Google GenAI integration.

The agent keeps its history in the Claude Messages format (see llm.py), so each call
translates that history into LangChain messages and translates Gemini's reply back.

Gemini 3 models attach a thought signature to each function call and reject a later
request that drops it. Every assistant message therefore also keeps Gemini's own
content and additional_kwargs under the "gemini" key, and they are replayed unchanged.

Reads GEMINI_API_KEY (or GOOGLE_API_KEY). TRIAGE_MODEL overrides the model.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from pipeline_triage.diagnosis import Diagnosis
from pipeline_triage.llm import Message, ModelRefusedError

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
# Free-tier models often return HTTP 503 (overloaded) for a few seconds. Five attempts
# with exponential backoff wait out a short spike (about 15 s of retries) and then fail,
# where the SDK's default of six attempts with delays up to 60 s looks like a hang.
MAX_ATTEMPTS = 5
_REFUSAL_FINISH_REASONS = frozenset({"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII"})


class GeminiTriageModel:
    def __init__(self, model: str | None = None, chat: BaseChatModel | None = None) -> None:
        if chat is None:
            from langchain_google_genai import ChatGoogleGenerativeAI

            chat = ChatGoogleGenerativeAI(
                model=model or os.environ.get("TRIAGE_MODEL", DEFAULT_GEMINI_MODEL),
                max_retries=MAX_ATTEMPTS,
            )
        self._chat = chat

    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        reply = await self._chat.bind_tools(tools).ainvoke(to_langchain(system, messages))
        if not isinstance(reply, AIMessage):
            raise TypeError(f"Expected an AIMessage, got {type(reply).__name__}.")
        _raise_on_refusal(reply)
        return from_langchain(reply)

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        structured = self._chat.with_structured_output(Diagnosis, include_raw=True)
        result = await structured.ainvoke(to_langchain(system, messages))
        if not isinstance(result, dict):
            raise TypeError(f"Expected raw and parsed output, got {type(result).__name__}.")
        if isinstance(result.get("raw"), AIMessage):
            _raise_on_refusal(result["raw"])
        parsed = result.get("parsed")
        if not isinstance(parsed, Diagnosis):
            raise RuntimeError(f"No diagnosis returned ({result.get('parsing_error')}).")
        return parsed


def to_langchain(system: str, messages: list[Message]) -> list[BaseMessage]:
    """Claude-format history to LangChain messages, with the system prompt first."""
    converted: list[BaseMessage] = [SystemMessage(system)]
    tool_names: dict[str, str] = {}
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            calls = [b for b in content if b.get("type") == "tool_use"]
            tool_names.update({c["id"]: c["name"] for c in calls})
            native = message.get("gemini", {})
            converted.append(
                AIMessage(
                    content=native.get("content", _text(content)),
                    additional_kwargs=native.get("additional_kwargs", {}),
                    tool_calls=[
                        {"name": c["name"], "args": c["input"], "id": c["id"]} for c in calls
                    ],
                )
            )
        elif isinstance(content, str):
            converted.append(HumanMessage(content))
        else:
            for block in content:
                if block.get("type") == "tool_result":
                    converted.append(
                        ToolMessage(
                            content=_text(block["content"]),
                            tool_call_id=block["tool_use_id"],
                            name=tool_names.get(block["tool_use_id"]),
                            status="error" if block.get("is_error") else "success",
                        )
                    )
                elif block.get("type") == "text":
                    converted.append(HumanMessage(block["text"]))
    return converted


def from_langchain(reply: AIMessage) -> Message:
    """A Gemini reply as a Claude-format assistant message, keeping Gemini's raw parts."""
    blocks: list[dict[str, Any]] = []
    if reply.text:
        blocks.append({"type": "text", "text": reply.text})
    for call in reply.tool_calls:
        if call["id"] is None:
            raise ValueError(f"Gemini returned a call to {call['name']} without an id.")
        blocks.append(
            {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
        )
    return {
        "role": "assistant",
        "content": blocks,
        "gemini": {"content": reply.content, "additional_kwargs": reply.additional_kwargs},
    }


def _text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if b.get("type") == "text")


def _raise_on_refusal(reply: AIMessage) -> None:
    reason = reply.response_metadata.get("finish_reason")
    if reason in _REFUSAL_FINISH_REASONS:
        raise ModelRefusedError(f"Gemini declined this request (finish reason: {reason}).")
