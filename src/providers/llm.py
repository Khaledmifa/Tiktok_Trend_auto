from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

import requests

from src.config import AppConfig

logger = logging.getLogger(__name__)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Extract JSON object from raw LLM text, stripping optional markdown fences."""
    cleaned = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, flags=re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1)
    else:
        brace_match = re.search(r"(\{.*\})", cleaned, flags=re.DOTALL)
        if brace_match:
            cleaned = brace_match.group(1)
    return json.loads(cleaned)


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate_json(
        self, prompt: str, system_prompt: str = "", contents: list[Any] | None = None
    ) -> dict[str, Any]:
        """Return parsed JSON dictionary from local text prompts."""


class LocalOllamaProvider(BaseLLMProvider):
    """100% Free Local Ollama provider ($0 cost, zero cloud API keys)."""

    def __init__(
        self,
        model: str,
        host: str = "http://127.0.0.1:11434",
        temperature: float = 0.7,
        fallback_models: list[str] | None = None,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.fallback_models = list(fallback_models or [])

    def _ordered_models(self) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for name in [self.model, *self.fallback_models]:
            clean = str(name or "").strip()
            if clean and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
        return ordered

    def generate_json(
        self, prompt: str, system_prompt: str = "", contents: list[Any] | None = None
    ) -> dict[str, Any]:
        user_msg = prompt
        if contents:
            text_parts = [str(c) for c in contents if isinstance(c, str)]
            if text_parts:
                user_msg = "\n".join(text_parts)

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_msg})

        last_exc: Exception | None = None
        for model_name in self._ordered_models():
            try:
                resp = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": model_name,
                        "messages": messages,
                        "format": "json",
                        "stream": False,
                        "options": {"temperature": self.temperature},
                    },
                    timeout=90,
                )
                resp.raise_for_status()
                raw_text = (resp.json().get("message") or {}).get("content") or "{}"
                return _extract_json_block(raw_text)
            except Exception as exc:
                last_exc = exc
                logger.debug("Local Ollama model '%s' skipped: %s", model_name, exc)

        assert last_exc is not None
        raise last_exc


def build_llm_provider(
    config: AppConfig, model_override: str | None = None, temperature: float | None = None
) -> BaseLLMProvider | None:
    """Return LocalOllamaProvider ONLY if a local Ollama server has an installed model; else None."""
    host = config.env("OLLAMA_HOST") or str(config.llm.get("ollama_host", "http://127.0.0.1:11434"))
    preferred_model = config.env("OLLAMA_MODEL") or model_override or str(config.llm.get("ollama_model", ""))
    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        if not resp.ok:
            return None
        models = [str(m.get("name", "")).strip() for m in resp.json().get("models", []) if m.get("name")]
        if not models:
            return None
        chosen = preferred_model if preferred_model in models else models[0]
        fallbacks = [m for m in models if m != chosen]
        temp = float(temperature if temperature is not None else config.llm.get("temperature", 0.7))
        return LocalOllamaProvider(
            model=chosen,
            host=host,
            temperature=temp,
            fallback_models=fallbacks,
        )
    except Exception:
        return None
