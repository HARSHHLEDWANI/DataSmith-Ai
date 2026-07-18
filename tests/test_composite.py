"""Tests for the composite-input features: detected-inputs manifest, PDF link
annotations, multi-source synthesis, and the YouTube audio fallback."""
from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path

import fitz
import pytest
from fastapi.testclient import TestClient


def _pdf_bytes(text: str, link_uri: str | None = None) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 742), text, fontsize=11)
    if link_uri:
        page.insert_link({"kind": fitz.LINK_URI, "from": fitz.Rect(50, 50, 200, 70), "uri": link_uri})
    data = doc.tobytes()
    doc.close()
    return data


def _wav_bytes() -> bytes:
    buf = io.BytesIO()
    framerate = 16000
    with wave.open(buf, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(framerate)
        frames = bytearray()
        for i in range(framerate):
            v = int(32767 * 0.3 * math.sin(2 * math.pi * 440 * i / framerate))
            frames += struct.pack("<h", v)
        w.writeframes(bytes(frames))
    return buf.getvalue()


def _app_client() -> TestClient:
    from app import main
    return TestClient(main.app)


def test_detected_inputs_manifest(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client

    PDF_TEXT = (
        "Recording from the last session:\n"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
        "Please review before the next meeting."
    )

    async def fake_plan_json(messages, **kw):
        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "plan": [{"step": 1, "tool": "qa", "input_from": "context", "reasoning": "answer"}],
        }

    async def fake_chat(messages, **kw):
        return "answer"

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "What video does this reference?", "conversation_id": "manifest1"},
        files=[("files", ("report.pdf", _pdf_bytes(PDF_TEXT), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    refs = body["detected_inputs"]
    assert refs, "expected the embedded YouTube link to be detected"
    assert refs[0]["kind"] == "youtube"
    assert refs[0]["value"] == "dQw4w9WgXcQ"
    assert refs[0]["found_in"] == "report.pdf"

    assert "embedded YouTube link" in body["input_manifest"]
    assert "report.pdf" in body["input_manifest"]


def test_manifest_reaches_planner_prompt(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client

    captured: dict = {}

    async def fake_plan_json(messages, **kw):
        captured["user"] = messages[-1]["content"]
        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "plan": [{"step": 1, "tool": "qa", "input_from": "context", "reasoning": "answer"}],
        }

    async def fake_chat(messages, **kw):
        return "answer"

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={
            "message": "Summarize the video at https://youtu.be/abcdefghijk please",
            "conversation_id": "manifest2",
        },
    )
    assert resp.status_code == 200
    assert "Detected inputs" in captured["user"]
    assert "YouTube" in captured["user"]


def test_pdf_link_annotations():
    from app.ingestion.pdf_extractor import extract_pdf
    from app.ingestion.references import detect_references

    data = _pdf_bytes(
        "Quarterly report with a clickable video link whose anchor text is not a URL.",
        link_uri="https://www.youtube.com/watch?v=A1b2C3d4E5f",
    )
    extracted = extract_pdf(data, "annotated.pdf")
    assert extracted.meta.get("link_uris") == ["https://www.youtube.com/watch?v=A1b2C3d4E5f"]

    refs = detect_references([extracted])
    assert len(refs) == 1
    assert refs[0].kind == "youtube"
    assert refs[0].via == "pdf_annotation"
    assert refs[0].found_in == "annotated.pdf"


def test_branching_plan_triggers_synthesis(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client

    PDF_TEXT = "The report says revenue grew 12% in Q2 and NPS reached 72."
    calls: list[str] = []

    async def fake_plan_json(messages, **kw):
        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "plan": [
                {"step": 1, "tool": "summarize", "input_from": "context", "reasoning": "summarize the pdf"},
                {"step": 2, "tool": "sentiment", "input_from": "context", "reasoning": "sentiment of the text"},
            ],
        }

    async def fake_chat(messages, **kw):
        calls.append(messages[0]["content"])
        if len(calls) == 1:
            return "SUMMARY OUTPUT"
        if len(calls) == 2:
            return "SENTIMENT OUTPUT"
        return "From the pdf (report.pdf): revenue grew.\n\nAcross sources: consistent."

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "Summarize this and give the sentiment", "conversation_id": "synth1"},
        files=[("files", ("report.pdf", _pdf_bytes(PDF_TEXT), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    tools = [s["tool"] for s in body["plan_trace"]]
    assert tools == ["summarize", "sentiment", "synthesize"]
    assert body["plan_trace"][-1]["status"] == "success"
    assert "Across sources" in body["final_answer"]
    assert len(calls) == 3  # two tool calls + one synthesis call


def test_linear_plan_skips_synthesis(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client
    from app.tools import youtube as yt

    PDF_TEXT = "Watch https://www.youtube.com/watch?v=dQw4w9WgXcQ before the meeting."
    chat_calls: list[int] = []

    async def fake_plan_json(messages, **kw):
        return {
            "needs_clarification": False,
            "clarifying_question": None,
            "plan": [
                {"step": 1, "tool": "youtube_transcript", "input_from": "context", "reasoning": "fetch"},
                {"step": 2, "tool": "summarize", "input_from": "step:1", "reasoning": "summarize"},
            ],
        }

    async def fake_fetch_transcript(video_id):
        return "the transcript"

    async def fake_chat(messages, **kw):
        chat_calls.append(1)
        return "THE SUMMARY"

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(yt, "fetch_transcript", fake_fetch_transcript)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "Summarize the linked video", "conversation_id": "synth2"},
        files=[("files", ("report.pdf", _pdf_bytes(PDF_TEXT), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    tools = [s["tool"] for s in body["plan_trace"]]
    assert "synthesize" not in tools
    assert body["final_answer"] == "THE SUMMARY"
    assert len(chat_calls) == 1  # no extra LLM call for single-terminal runs


@pytest.mark.asyncio
async def test_yt_audio_fallback_success(monkeypatch, tmp_path):
    from app.config import settings
    from app.ingestion import audio_extractor
    from app.tools import youtube as yt

    def fake_fetch_sync(video_id):
        return "[No transcript available for this video — captions are disabled or missing.]"

    def fake_download(video_id, tmpdir):
        path = Path(tmpdir) / f"{video_id}.webm"
        path.write_bytes(b"fake-audio-bytes")
        return str(path), None

    async def fake_transcribe(data, source):
        assert data == b"fake-audio-bytes"
        return "WHISPER TRANSCRIPT"

    monkeypatch.setattr(yt, "_fetch_sync", fake_fetch_sync)
    monkeypatch.setattr(yt, "_download_audio_sync", fake_download)
    monkeypatch.setattr(audio_extractor, "transcribe_bytes", fake_transcribe)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "yt_audio_fallback", True)

    result = await yt.fetch_transcript("dQw4w9WgXcQ")
    assert result.startswith("(transcript unavailable — transcribed from audio via Whisper)")
    assert "WHISPER TRANSCRIPT" in result


@pytest.mark.asyncio
async def test_yt_audio_fallback_disabled(monkeypatch):
    from app.config import settings
    from app.tools import youtube as yt

    CAPTIONS_ERROR = "[No transcript available for this video — captions are disabled or missing.]"

    def fake_fetch_sync(video_id):
        return CAPTIONS_ERROR

    monkeypatch.setattr(yt, "_fetch_sync", fake_fetch_sync)
    monkeypatch.setattr(settings, "yt_audio_fallback", False)

    result = await yt.fetch_transcript("dQw4w9WgXcQ")
    assert result == CAPTIONS_ERROR


@pytest.mark.asyncio
async def test_yt_audio_fallback_download_fails(monkeypatch):
    from app.config import settings
    from app.tools import youtube as yt

    def fake_fetch_sync(video_id):
        return "[No transcript available for this video — captions are disabled or missing.]"

    def fake_download(video_id, tmpdir):
        return None, "audio download failed: blocked"

    monkeypatch.setattr(yt, "_fetch_sync", fake_fetch_sync)
    monkeypatch.setattr(yt, "_download_audio_sync", fake_download)
    monkeypatch.setattr(settings, "groq_api_key", "test-key")
    monkeypatch.setattr(settings, "yt_audio_fallback", True)

    result = await yt.fetch_transcript("dQw4w9WgXcQ")
    assert result.startswith("[No transcript available")
    assert "Audio-download fallback also failed" in result
