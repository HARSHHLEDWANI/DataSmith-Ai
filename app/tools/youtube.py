from __future__ import annotations

import asyncio
import re

from app.logging_config import get_logger
from app.tools.registry import Tool, ToolContext, registry

logger = get_logger("tools.youtube")

_YT_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^&\s]*&)*v=|shorts/|embed/|v/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)


def find_youtube_urls(text: str) -> list[str]:
    seen: list[str] = []
    for match in _YT_RE.finditer(text or ""):
        vid = match.group(1)
        if vid not in seen:
            seen.append(vid)
    return seen


def _fetch_sync(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api._errors import (
            NoTranscriptFound,
            TranscriptsDisabled,
            VideoUnavailable,
        )
    except Exception as exc:
        return f"[YouTube transcript unavailable: {exc}]"

    try:
        if hasattr(YouTubeTranscriptApi, "get_transcript"):
            entries = YouTubeTranscriptApi.get_transcript(video_id)
        else:
            entries = YouTubeTranscriptApi().fetch(video_id).to_raw_data()
        text = " ".join(part.get("text", "") for part in entries).strip()
        return text or "[Transcript was empty for this video.]"
    except (TranscriptsDisabled, NoTranscriptFound):
        return "[No transcript available for this video — captions are disabled or missing.]"
    except VideoUnavailable:
        return "[The referenced YouTube video is unavailable or private.]"
    except Exception as exc:
        msg = str(exc).lower()
        if "no element found" in msg or "parseerror" in msg or "xml" in msg:
            return "[No transcript available — YouTube returned an empty response (captions disabled, live stream, or the server IP is blocked).]"
        return f"[Could not fetch transcript: {exc}]"


async def fetch_transcript(video_id: str) -> str:
    for attempt in range(2):
        result = await asyncio.to_thread(_fetch_sync, video_id)
        if not result.startswith("[Could not fetch transcript") or attempt == 1:
            return result
        await asyncio.sleep(0.6)
    return result


async def youtube_transcript(ctx: ToolContext) -> str:
    haystack = "\n".join([ctx.query, ctx.upstream, ctx.combined_text()])
    video_ids = find_youtube_urls(haystack)
    if not video_ids:
        return "[No YouTube URL was found in the query or any uploaded input.]"
    logger.info("fetching youtube transcripts", extra={"data": {"video_ids": video_ids}})
    chunks = []
    for vid in video_ids:
        transcript = await fetch_transcript(vid)
        chunks.append(f"Transcript for video {vid}:\n{transcript}")
    return "\n\n".join(chunks)


registry.register(
    Tool(
        name="youtube_transcript",
        description=(
            "Detect a YouTube URL anywhere in the query OR any uploaded input "
            "(e.g. a link inside a PDF) and fetch its transcript. Returns a "
            "message if captions are unavailable. Chain summarize/qa after this."
        ),
        input_hint="text containing a YouTube URL (auto-detected)",
        func=youtube_transcript,
    )
)
