from __future__ import annotations

from app.config import settings
from app.ingestion.audio_extractor import extract_audio
from app.ingestion.image_extractor import extract_image
from app.ingestion.models import ExtractedInput
from app.ingestion.pdf_extractor import extract_pdf
from app.ingestion.text_extractor import extract_text
from app.logging_config import get_logger

logger = get_logger("ingestion.router")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif", ".webp"}
PDF_EXTS = {".pdf"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}
TEXT_EXTS = {".txt", ".md", ".csv", ".log", ".json"}


def detect_type(source: str, content_type: str | None) -> str:
    lower = source.lower()
    ext = lower[lower.rfind(".") :] if "." in lower else ""
    if ext in IMAGE_EXTS:
        return "image"
    if ext in PDF_EXTS:
        return "pdf"
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in TEXT_EXTS:
        return "text"
    if content_type:
        if content_type.startswith("image/"):
            return "image"
        if content_type == "application/pdf":
            return "pdf"
        if content_type.startswith("audio/") or content_type.startswith("video/"):
            return "audio"
        if content_type.startswith("text/"):
            return "text"
    return "unknown"


async def ingest_file(
    data: bytes, source: str, content_type: str | None
) -> ExtractedInput:
    if len(data) == 0:
        return ExtractedInput(source=source, type="unknown", error="Empty file.")
    if len(data) > settings.max_file_size_bytes:
        return ExtractedInput(
            source=source,
            type="unknown",
            error=f"File exceeds {settings.max_file_size_mb}MB limit.",
        )

    file_type = detect_type(source, content_type)
    logger.info(
        "ingesting file",
        extra={"data": {"source": source, "type": file_type, "bytes": len(data)}},
    )

    if file_type == "image":
        return extract_image(data, source)
    if file_type == "pdf":
        return extract_pdf(data, source)
    if file_type == "audio":
        return await extract_audio(data, source)
    if file_type == "text":
        return extract_text(data, source)

    return ExtractedInput(
        source=source,
        type="unknown",
        error=f"Unsupported file type for '{source}'. Supported: images, PDF, audio, text.",
    )
