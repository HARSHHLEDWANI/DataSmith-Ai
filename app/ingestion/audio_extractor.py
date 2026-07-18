from __future__ import annotations

import asyncio
import io

import httpx
from mutagen import File as MutagenFile

from app.config import settings
from app.ingestion.models import ExtractedInput
from app.ingestion.text_extractor import normalize_text
from app.logging_config import get_logger

logger = get_logger("ingestion.audio")


def _duration_seconds(data: bytes, source: str) -> float | None:
    try:
        buffer = io.BytesIO(data)
        buffer.name = source
        audio = MutagenFile(buffer)
        if audio is not None and audio.info is not None:
            return round(float(audio.info.length), 2)
    except Exception as exc:
        logger.warning("duration probe failed", extra={"data": {"source": source, "error": str(exc)}})
    return None


async def transcribe_bytes(data: bytes, source: str) -> str:
    url = settings.groq_base_url.rstrip("/") + "/audio/transcriptions"
    headers = {"Authorization": f"Bearer {settings.whisper_key()}"}
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=settings.request_timeout_s) as client:
                resp = await client.post(
                    url,
                    headers=headers,
                    data={"model": settings.whisper_model, "response_format": "json"},
                    files={"file": (source, data)},
                )
                resp.raise_for_status()
                return resp.json().get("text", "")
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "whisper call failed",
                extra={"data": {"source": source, "attempt": attempt, "error": str(exc)}},
            )
            if attempt == 0:
                await asyncio.sleep(1.0)
    raise RuntimeError(f"transcription failed after retry: {last_exc}")


_transcribe = transcribe_bytes


async def extract_audio(data: bytes, source: str) -> ExtractedInput:
    duration = _duration_seconds(data, source)
    meta: dict[str, object] = {}
    if duration is not None:
        meta["duration_seconds"] = duration

    if not settings.whisper_configured():
        return ExtractedInput(
            source=source,
            type="audio",
            meta=meta,
            error="Audio transcription unavailable: set GROQ_API_KEY for Whisper.",
        )

    try:
        text = normalize_text(await transcribe_bytes(data, source))
    except Exception as exc:
        return ExtractedInput(source=source, type="audio", meta=meta, error=f"Transcription failed: {exc}")

    meta["chars"] = len(text)
    return ExtractedInput(source=source, type="audio", text=text, meta=meta)
