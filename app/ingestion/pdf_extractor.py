from __future__ import annotations

import io

import fitz

from app.ingestion.image_extractor import ocr_image
from app.ingestion.models import ExtractedInput
from app.ingestion.text_extractor import normalize_text
from app.logging_config import get_logger

logger = get_logger("ingestion.pdf")

MIN_TEXT_CHARS = 30
OCR_RENDER_DPI = 200


def extract_pdf(data: bytes, source: str) -> ExtractedInput:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        logger.warning("pdf open failed", extra={"data": {"source": source, "error": str(exc)}})
        return ExtractedInput(source=source, type="pdf", error=f"Unreadable PDF: {exc}")

    page_texts: list[str] = []
    ocr_pages: list[int] = []
    ocr_confidences: list[float] = []

    try:
        from PIL import Image

        for index, page in enumerate(doc):
            page_num = index + 1
            text = (page.get_text() or "").strip()
            if len(text) < MIN_TEXT_CHARS:
                try:
                    matrix = fitz.Matrix(OCR_RENDER_DPI / 72, OCR_RENDER_DPI / 72)
                    pix = page.get_pixmap(matrix=matrix)
                    image = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text, conf = ocr_image(image)
                    if ocr_text.strip():
                        text = ocr_text
                        ocr_pages.append(page_num)
                        ocr_confidences.append(conf)
                except Exception as exc:
                    logger.warning(
                        "pdf page ocr failed",
                        extra={"data": {"source": source, "page": page_num, "error": str(exc)}},
                    )
            page_texts.append(text)
    finally:
        doc.close()

    full_text = normalize_text("\n\n".join(page_texts))
    mean_conf = (
        round(sum(ocr_confidences) / len(ocr_confidences), 2) if ocr_confidences else None
    )

    meta: dict[str, object] = {"pages": len(page_texts), "ocr_pages": ocr_pages, "chars": len(full_text)}
    if mean_conf is not None:
        meta["ocr_confidence"] = mean_conf

    if not full_text.strip():
        return ExtractedInput(
            source=source,
            type="pdf",
            meta=meta,
            error="No extractable text found (empty or image-only with failed OCR).",
        )

    return ExtractedInput(source=source, type="pdf", text=full_text, meta=meta)
