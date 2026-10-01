"""Provider-neutral LLM factory. No database-specific prompt logic lives here."""
from __future__ import annotations

from agent.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_PROVIDER,
    OLLAMA_MODEL,
    OLLAMA_URL,
    OPEN_ROUTER_API_KEY,
    OPEN_ROUTER_BASE_URL,
    OPEN_ROUTER_MODEL,
)
from agent.token_usage import get_token_usage_callback

_LLM = None


def get_llm():
    global _LLM
    if _LLM is not None:
        return _LLM

    callback = get_token_usage_callback()
    callbacks = [callback] if callback else None

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required")
        _LLM = ChatAnthropic(
            model=ANTHROPIC_MODEL,
            api_key=ANTHROPIC_API_KEY,
            temperature=0,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            callbacks=callbacks,
        )
    elif LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama
        _LLM = ChatOllama(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            temperature=0,
            num_ctx=8192,
            num_predict=LLM_MAX_OUTPUT_TOKENS,
            callbacks=callbacks,
        )
    else:
        from langchain_openai import ChatOpenAI
        if not OPEN_ROUTER_API_KEY:
            raise ValueError("OPEN_ROUTER_API_KEY is required when LLM_PROVIDER=openrouter")
        _LLM = ChatOpenAI(
            model=OPEN_ROUTER_MODEL,
            api_key=OPEN_ROUTER_API_KEY,
            base_url=OPEN_ROUTER_BASE_URL,
            temperature=0,
            max_tokens=LLM_MAX_OUTPUT_TOKENS,
            callbacks=callbacks,
        )
    return _LLM
