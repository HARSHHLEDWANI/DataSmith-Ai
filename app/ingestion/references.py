"""Detect references (YouTube links, URLs) embedded inside extracted inputs.

Lives in the ingestion layer so both tools and the agent can import it
without creating an import cycle.
"""
from __future__ import annotations

import re

from pydantic import BaseModel

from app.ingestion.models import ExtractedInput

YT_RE = re.compile(
    r"(?:https?://)?(?:www\.|m\.)?"
    r"(?:youtube\.com/(?:watch\?(?:[^&\s]*&)*v=|shorts/|embed/|v/|live/)|youtu\.be/)"
    r"([A-Za-z0-9_-]{11})",
    re.IGNORECASE,
)

URL_RE = re.compile(r"https?://[^\s<>\"'\)\]\}]+", re.IGNORECASE)

MAX_REFS_PER_INPUT = 20

_TYPE_LABELS = {"pdf": "PDF", "image": "image", "audio": "audio file", "text": "text file"}


class DetectedReference(BaseModel):
    kind: str  # "youtube" | "url"
    value: str  # video id for youtube, full url otherwise
    url: str  # canonical url
    found_in: str  # ExtractedInput.source, e.g. "report.pdf" or "user_query"
    via: str = "text"  # "text" | "pdf_annotation"


def find_youtube_ids(text: str) -> list[str]:
    seen: list[str] = []
    for match in YT_RE.finditer(text or ""):
        vid = match.group(1)
        if vid not in seen:
            seen.append(vid)
    return seen


def _scan_text(text: str, source: str, via: str) -> list[DetectedReference]:
    refs: list[DetectedReference] = []
    for vid in find_youtube_ids(text):
        refs.append(
            DetectedReference(
                kind="youtube",
                value=vid,
                url=f"https://www.youtube.com/watch?v={vid}",
                found_in=source,
                via=via,
            )
        )
    for match in URL_RE.finditer(text or ""):
        url = match.group(0).rstrip(".,;:")
        if YT_RE.search(url):
            continue  # already captured as a youtube ref
        refs.append(
            DetectedReference(kind="url", value=url, url=url, found_in=source, via=via)
        )
    return refs[:MAX_REFS_PER_INPUT]


def detect_references(inputs: list[ExtractedInput]) -> list[DetectedReference]:
    """Scan every input's text (and PDF link annotations) for embedded references."""
    refs: list[DetectedReference] = []
    seen: set[tuple[str, str]] = set()
    for inp in inputs:
        candidates = _scan_text(inp.text, inp.source, via="text")
        for uri in inp.meta.get("link_uris", []) or []:
            candidates.extend(_scan_text(uri, inp.source, via="pdf_annotation"))
        for ref in candidates:
            key = (ref.kind, ref.value)
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def describe_manifest(
    inputs: list[ExtractedInput], refs: list[DetectedReference]
) -> str:
    """One-line inventory of everything detected in this request, e.g.

    "Detected inputs: 1 PDF (report.pdf) — contains 1 embedded YouTube link;
     1 image (chart.png); user question."
    """
    if not inputs:
        return "Detected inputs: none."

    refs_by_source: dict[str, list[DetectedReference]] = {}
    for ref in refs:
        refs_by_source.setdefault(ref.found_in, []).append(ref)

    parts: list[str] = []
    for inp in inputs:
        if inp.source == "user_query":
            label = "user question"
        else:
            label = f"1 {_TYPE_LABELS.get(inp.type, inp.type)} ({inp.source})"
        if inp.error:
            label += " [extraction failed]"
        embedded = refs_by_source.get(inp.source, [])
        if embedded:
            yt = sum(1 for r in embedded if r.kind == "youtube")
            other = len(embedded) - yt
            bits = []
            if yt:
                bits.append(f"{yt} embedded YouTube link{'s' if yt > 1 else ''}")
            if other:
                bits.append(f"{other} embedded link{'s' if other > 1 else ''}")
            label += f" — contains {' and '.join(bits)}"
        parts.append(label)
    return "Detected inputs: " + "; ".join(parts) + "."
