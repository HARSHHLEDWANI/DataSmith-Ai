from __future__ import annotations

import io

import pytesseract
from PIL import Image, ImageOps

from app.ingestion.models import ExtractedInput
from app.ingestion.text_extractor import normalize_text
from app.logging_config import get_logger

logger = get_logger("ingestion.image")


def _ocr_confidence(image: Image.Image) -> tuple[str, float]:
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[str] = []
    confidences: list[float] = []
    for word, conf in zip(data.get("text", []), data.get("conf", [])):
        if word and word.strip():
            words.append(word)
            try:
                c = float(conf)
            except (TypeError, ValueError):
                c = -1.0
            if c >= 0:
                confidences.append(c)
    text = " ".join(words)
    mean_conf = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    return text, mean_conf


def ocr_image(image: Image.Image) -> tuple[str, float]:
    prepared = ImageOps.autocontrast(ImageOps.grayscale(image))
    text, conf = _ocr_confidence(prepared)
    return normalize_text(text), conf


def extract_image(data: bytes, source: str) -> ExtractedInput:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        logger.warning("image open failed", extra={"data": {"source": source, "error": str(exc)}})
        return ExtractedInput(source=source, type="image", error=f"Unreadable image: {exc}")

    try:
        text, conf = ocr_image(image)
    except pytesseract.TesseractNotFoundError:
        return ExtractedInput(
            source=source,
            type="image",
            error="Tesseract OCR engine not installed on the server.",
        )
    except Exception as exc:
        return ExtractedInput(source=source, type="image", error=f"OCR failed: {exc}")

    return ExtractedInput(
        source=source,
        type="image",
        text=text,
        meta={"ocr_confidence": conf, "width": image.width, "height": image.height, "chars": len(text)},
    )
