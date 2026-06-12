from __future__ import annotations

from app.ingestion.models import ExtractedInput


def normalize_text(raw: str) -> str:
    lines = [line.rstrip() for line in raw.replace("\r\n", "\n").split("\n")]
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            if blanks <= 1:
                out.append("")
        else:
            blanks = 0
            out.append(line)
    return "\n".join(out).strip()


def extract_text(data: bytes, source: str) -> ExtractedInput:
    try:
        decoded = data.decode("utf-8")
    except UnicodeDecodeError:
        decoded = data.decode("latin-1", errors="replace")
    text = normalize_text(decoded)
    return ExtractedInput(source=source, type="text", text=text, meta={"chars": len(text)})


def extract_plain(text: str, source: str = "user_query") -> ExtractedInput:
    norm = normalize_text(text)
    return ExtractedInput(source=source, type="text", text=norm, meta={"chars": len(norm)})
