"""Gemini and Groq behind the TriageModel interface, through LangChain chat models.

Both have free tiers. `LangChainTriageModel` wraps any LangChain chat model;
`gemini_model()` and `groq_model()` build one for each provider.

The agent keeps its history in the Claude Messages format (see llm.py), so each call
translates that history into LangChain messages and translates the reply back.

A reply can carry provider data that must come back unchanged on the next request.
Gemini 3 attaches a thought signature to each function call and rejects a later
request that drops it. Every assistant message therefore also keeps the provider's
own content and additional_kwargs under the "native" key, and they are replayed as-is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

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
    from langchain_core.runnables import Runnable

DEFAULT_MODELS = {"gemini": "gemini-3.5-flash", "groq": "openai/gpt-oss-120b"}

# Gemini's free-tier models often return HTTP 503 (overloaded) for a few seconds. Five
# attempts with exponential backoff wait out a short spike (about 15 s of retries) and
# then fail, where the SDK's default of six attempts with delays up to 60 s looks like
# a hang.
GEMINI_MAX_ATTEMPTS = 5
# Groq's free tier allows 8,000 tokens a minute, and every round resends the whole
# investigation, so later rounds often get HTTP 429. The SDK waits for the retry-after
# the API sends, so six retries sit out the rest of the minute.
GROQ_MAX_RETRIES = 6
# How many times to ask the model again after the provider rejects its tool call.
TOOL_CALL_REPAIRS = 2

# Gemini's safety stops, and the OpenAI-style one Groq uses.
_REFUSAL_FINISH_REASONS = frozenset(
    {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "content_filter"}
)

StructuredMethod = Literal["function_calling", "json_schema"]


class LangChainTriageModel:
    def __init__(self, chat: BaseChatModel, structured_method: StructuredMethod) -> None:
        self._chat = chat
        self._structured_method = structured_method

    async def next_turn(
        self, system: str, messages: list[Message], tools: list[dict[str, Any]]
    ) -> Message:
        bound = self._chat.bind_tools(tools)
        reply = await _invoke_with_repair(bound, to_langchain(system, messages))
        if not isinstance(reply, AIMessage):
            raise TypeError(f"Expected an AIMessage, got {type(reply).__name__}.")
        _raise_on_refusal(reply)
        return from_langchain(reply)

    async def diagnose(self, system: str, messages: list[Message]) -> Diagnosis:
        structured = self._chat.with_structured_output(
            Diagnosis, method=self._structured_method, include_raw=True
        )
        result = await _invoke_with_repair(structured, to_langchain(system, messages))
        if not isinstance(result, dict):
            raise TypeError(f"Expected raw and parsed output, got {type(result).__name__}.")
        if isinstance(result.get("raw"), AIMessage):
            _raise_on_refusal(result["raw"])
        parsed = result.get("parsed")
        if not isinstance(parsed, Diagnosis):
            raise RuntimeError(f"No diagnosis returned ({result.get('parsing_error')}).")
        return parsed


async def _invoke_with_repair(runnable: Runnable[Any, Any], prompt: list[BaseMessage]) -> Any:
    """Invoke, and when the provider rejects the model's tool call, say why and ask again.

    Groq validates tool calls itself and rejects the whole reply, so a call to a tool
    that is not offered, or since="2026-10-28" for a date-time, never reaches the MCP
    server, which would otherwise hand the error back to the model. The note is sent
    only with the retry and never added to the history.
    """
    for attempt in range(TOOL_CALL_REPAIRS + 1):
        try:
            return await runnable.ainvoke(prompt)
        except Exception as error:
            problem = _rejected_tool_call(error)
            if problem is None or attempt == TOOL_CALL_REPAIRS:
                raise
            note = (
                f"Your last tool call was rejected: {problem}\n"
                "Only call the tools you were given, with arguments that match their schemas."
            )
            prompt = [*prompt, HumanMessage(note)]
    raise AssertionError("unreachable")


def _rejected_tool_call(error: Exception) -> str | None:
    """Why the provider rejected the model's tool call, or None for any other error."""
    import groq

    if isinstance(error, groq.BadRequestError) and isinstance(error.body, dict):
        details = error.body.get("error", error.body)
        if isinstance(details, dict) and details.get("code") == "tool_use_failed":
            return str(details.get("message", "the arguments did not match the schema"))
    return None


def gemini_model(model: str | None = None) -> LangChainTriageModel:
    """Gemini. Reads GEMINI_API_KEY (or GOOGLE_API_KEY)."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    chat = ChatGoogleGenerativeAI(
        model=model or DEFAULT_MODELS["gemini"], max_retries=GEMINI_MAX_ATTEMPTS
    )
    # Gemini constrains the response to the JSON schema natively.
    return LangChainTriageModel(chat, structured_method="json_schema")


def groq_model(model: str | None = None) -> LangChainTriageModel:
    """Open-weight models on Groq. Reads GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    chat = ChatGroq(model=model or DEFAULT_MODELS["groq"], max_retries=GROQ_MAX_RETRIES)
    # Not every Groq model supports JSON schema output; every tool-calling model can
    # return the diagnosis as a forced tool call.
    return LangChainTriageModel(chat, structured_method="function_calling")


def to_langchain(system: str, messages: list[Message]) -> list[BaseMessage]:
    """Claude-format history to LangChain messages, with the system prompt first."""
    converted: list[BaseMessage] = [SystemMessage(system)]
    tool_names: dict[str, str] = {}
    for message in messages:
        content = message["content"]
        if message["role"] == "assistant":
            calls = [b for b in content if b.get("type") == "tool_use"]
            tool_names.update({c["id"]: c["name"] for c in calls})
            native = message.get("native", {})
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
    """A LangChain reply as a Claude-format assistant message, keeping the raw parts."""
    blocks: list[dict[str, Any]] = []
    if reply.text:
        blocks.append({"type": "text", "text": reply.text})
    for call in reply.tool_calls:
        if call["id"] is None:
            raise ValueError(f"The model returned a call to {call['name']} without an id.")
        blocks.append(
            {"type": "tool_use", "id": call["id"], "name": call["name"], "input": call["args"]}
        )
    return {
        "role": "assistant",
        "content": blocks,
        "native": {"content": reply.content, "additional_kwargs": reply.additional_kwargs},
    }


def _text(content: str | list[dict[str, Any]]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b["text"] for b in content if b.get("type") == "text")


def _raise_on_refusal(reply: AIMessage) -> None:
    reason = reply.response_metadata.get("finish_reason")
    if reason in _REFUSAL_FINISH_REASONS:
        raise ModelRefusedError(f"The model declined this request (finish reason: {reason}).")
