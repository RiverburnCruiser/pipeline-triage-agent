"""Plain-language messages for errors a user can fix: a missing key, a used-up quota.

Shared by the CLI, the HTTP API and the agent eval.
"""

from __future__ import annotations

import groq
from anthropic import AuthenticationError
from langchain_core.exceptions import ModelError

from pipeline_triage.llm import ModelRefusedError

_HINTS = {
    ("gemini", 429): "This model's quota is used up. Gemini's free tier limits requests per"
    " minute and per day for each model, so wait, or switch with TRIAGE_MODEL (see README).",
    ("groq", 429): "Groq's free-tier limit was reached: tokens per minute, or requests per day."
    " Wait a minute and try again, or switch with TRIAGE_MODEL (see README).",
    ("groq", 413): "The request is larger than Groq's free-tier tokens-per-minute limit."
    " Try a model with a higher limit (see README).",
    ("gemini", 503): "The model is overloaded. Try again, or switch with TRIAGE_MODEL.",
    ("groq", 503): "The model is overloaded. Try again, or switch with TRIAGE_MODEL.",
}


def _api_error(provider: str, code: object, message: str) -> str:
    hint = _HINTS.get((provider, code), "") if isinstance(code, int) else ""
    first_line = message.splitlines()[0][:300] if message else "no details"
    return f"{provider.capitalize()} request failed: {first_line}\n{hint}".strip()


def explain_error(error: BaseException) -> str | None:
    """A one-line message and next step for an error the user can fix, or None."""
    message = str(error)
    if isinstance(error, AuthenticationError):
        return "Claude rejected the credentials. Check ANTHROPIC_API_KEY (see README)."
    if isinstance(error, TypeError) and "authentication method" in message:
        # Raised by the Anthropic SDK when it finds no credentials at all.
        return "No Claude credentials found. Set ANTHROPIC_API_KEY (see README)."
    if isinstance(error, ModelRefusedError):
        return message
    if isinstance(error, groq.AuthenticationError):
        return "Groq rejected the API key. Check GROQ_API_KEY (see README)."
    if isinstance(error, groq.APIStatusError):
        return _api_error("groq", error.status_code, message)
    if isinstance(error, groq.GroqError) and "api_key" in message:
        return "No Groq API key found. Set GROQ_API_KEY (see README)."
    if isinstance(error, ModelError):
        # LangChain's base class for Gemini's errors. The HTTP status is on the
        # underlying google-genai error.
        if "API key not valid" in message:
            return "Gemini rejected the API key. Check GEMINI_API_KEY (see README)."
        code = getattr(error, "code", None) or getattr(error.__cause__, "code", None)
        return _api_error("gemini", code, message)
    if isinstance(error, ValueError):
        if "API key required" in message:
            return "No Gemini API key found. Set GEMINI_API_KEY (see README)."
        if "TRIAGE_PROVIDER" in message:
            return message
    return None
