from __future__ import annotations

import asyncio
import json
import re
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from app.config import settings
from app.logging_config import get_logger

logger = get_logger("llm.client")

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class LLMError(RuntimeError):
    pass


class LLMNotConfiguredError(LLMError):
    pass


def _strip_fences(text: str) -> str:
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else text.strip()


def _extract_json_object(text: str) -> str:
    cleaned = _strip_fences(text)
    start = cleaned.find("{")
    if start == -1:
        return cleaned
    depth = 0
    for i in range(start, len(cleaned)):
        if cleaned[i] == "{":
            depth += 1
        elif cleaned[i] == "}":
            depth -= 1
            if depth == 0:
                return cleaned[start : i + 1]
    return cleaned[start:]


class LLMClient:

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def _ensure_client(self) -> AsyncOpenAI:
        if not settings.llm_configured():
            raise LLMNotConfiguredError(
                "LLM_API_KEY is not set. Set LLM_API_KEY (and optionally "
                "LLM_BASE_URL / LLM_MODEL) to enable the agent."
            )
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                timeout=settings.llm_timeout_s,
                max_retries=0,
            )
        return self._client

    async def _with_retry(self, coro_factory, *, what: str) -> Any:
        last_exc: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    coro_factory(), timeout=settings.llm_timeout_s
                )
            except LLMNotConfiguredError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "llm call failed",
                    extra={"data": {"what": what, "attempt": attempt, "error": str(exc)}},
                )
                if attempt == 0:
                    await asyncio.sleep(0.8 * (attempt + 1))
        raise LLMError(f"{what} failed after retry: {last_exc}") from last_exc

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> str:
        client = self._ensure_client()

        async def _call() -> str:
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content or ""

        return await self._with_retry(_call, what="chat")

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 1500,
    ) -> dict[str, Any]:
        client = self._ensure_client()

        async def _call(msgs: list[dict[str, str]]) -> str:
            resp = await client.chat.completions.create(
                model=settings.llm_model,
                messages=msgs,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content or ""

        raw = await self._with_retry(lambda: _call(messages), what="chat_json")
        try:
            return json.loads(_extract_json_object(raw))
        except (json.JSONDecodeError, ValueError):
            logger.warning("invalid json from llm, re-prompting once")
            repair_messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. Respond with "
                        "ONLY a single valid JSON object, no prose, no markdown."
                    ),
                },
            ]
            raw2 = await self._with_retry(
                lambda: _call(repair_messages), what="chat_json_repair"
            )
            try:
                return json.loads(_extract_json_object(raw2))
            except (json.JSONDecodeError, ValueError) as exc:
                raise LLMError(f"LLM did not return valid JSON: {exc}") from exc

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> AsyncIterator[str]:
        client = self._ensure_client()
        try:
            stream = await client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta
        except LLMNotConfiguredError:
            raise
        except Exception as exc:
            raise LLMError(f"streaming failed: {exc}") from exc


llm_client = LLMClient()
