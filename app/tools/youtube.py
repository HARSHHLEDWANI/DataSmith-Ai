from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from app.config import settings
from app.ingestion.references import YT_RE as _YT_RE  # noqa: F401  (kept for compat)
from app.ingestion.references import find_youtube_ids as find_youtube_urls
from app.logging_config import get_logger
from app.tools.registry import Tool, ToolContext, registry

logger = get_logger("tools.youtube")


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


def _download_audio_sync(video_id: str, tmpdir: str) -> tuple[str | None, str | None]:
    """Download the smallest audio stream to tmpdir. Returns (path, None) or (None, error)."""
    try:
        import yt_dlp
    except Exception as exc:
        return None, f"yt-dlp is not installed: {exc}"

    url = f"https://www.youtube.com/watch?v={video_id}"
    opts = {
        "format": "worstaudio/worst",
        "outtmpl": os.path.join(tmpdir, "%(id)s.%(ext)s"),
        "max_filesize": settings.yt_fallback_max_bytes,
        "quiet": True,
        "noprogress": True,
        "noplaylist": True,
        "socket_timeout": 30,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get("duration") or 0
            if duration > settings.yt_fallback_max_duration_s:
                return None, (
                    f"video is {duration}s long, over the "
                    f"{settings.yt_fallback_max_duration_s}s audio-fallback cap"
                )
            ydl.download([url])
    except Exception as exc:
        return None, f"audio download failed: {exc}"

    for entry in Path(tmpdir).iterdir():
        if entry.is_file() and entry.stat().st_size:
            if entry.stat().st_size > settings.yt_fallback_max_bytes:
                return None, "downloaded audio exceeds the transcription size cap"
            return str(entry), None
    return None, "audio download produced no file"


async def _fallback_transcribe(video_id: str) -> str | None:
    """Captions failed — download the audio and transcribe via Groq Whisper.
    Returns transcript text, or None if the fallback itself failed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path, err = await asyncio.to_thread(_download_audio_sync, video_id, tmpdir)
        if err:
            logger.warning("yt audio fallback failed", extra={"data": {"video_id": video_id, "error": err}})
            return None
        data = Path(path).read_bytes()
        try:
            from app.ingestion.audio_extractor import transcribe_bytes

            text = (await transcribe_bytes(data, Path(path).name)).strip()
        except Exception as exc:
            logger.warning(
                "yt fallback transcription failed",
                extra={"data": {"video_id": video_id, "error": str(exc)}},
            )
            return None
    return text or None


async def fetch_transcript(video_id: str) -> str:
    for attempt in range(2):
        result = await asyncio.to_thread(_fetch_sync, video_id)
        if not result.startswith("[Could not fetch transcript") or attempt == 1:
            break
        await asyncio.sleep(0.6)

    if result.startswith("[") and settings.yt_audio_fallback and settings.whisper_configured():
        logger.info("captions unavailable, trying audio fallback", extra={"data": {"video_id": video_id}})
        fallback = await _fallback_transcribe(video_id)
        if fallback:
            return "(transcript unavailable — transcribed from audio via Whisper)\n" + fallback
        return result[:-1] + " Audio-download fallback also failed.]"
    return result


async def youtube_transcript(ctx: ToolContext) -> str:
    annotation_uris = [
        uri for inp in ctx.inputs for uri in (inp.meta.get("link_uris") or [])
    ]
    haystack = "\n".join([ctx.query, ctx.upstream, ctx.combined_text(), *annotation_uris])
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
            "(e.g. a link inside a PDF) and fetch its transcript. Falls back to "
            "downloading the audio and transcribing it via Whisper when captions "
            "are unavailable. Chain summarize/qa after this."
        ),
        input_hint="text containing a YouTube URL (auto-detected)",
        func=youtube_transcript,
    )
)
