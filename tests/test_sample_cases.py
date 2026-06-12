from __future__ import annotations

import io
import math
import struct
import wave

import fitz
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.ingestion.models import ExtractedInput


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


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 742), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _png_bytes() -> bytes:
    img = Image.new("RGB", (640, 160), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _plan(*tools: str) -> dict:
    steps = []
    for i, tool in enumerate(tools):
        steps.append({
            "step": i + 1,
            "tool": tool,
            "input_from": "context" if i == 0 else f"step:{i}",
            "reasoning": tool,
        })
    return {"needs_clarification": False, "clarifying_question": None, "plan": steps}


def _app_client() -> TestClient:
    from app import main
    return TestClient(main.app)


def test_tc1_audio_transcription_and_summary(monkeypatch):
    from app.agent import planner
    from app.ingestion import router as ir
    from app.llm.client import llm_client

    TRANSCRIPT = (
        "Welcome to this lecture on deep learning. We will cover gradient descent, "
        "backpropagation, and neural network architectures in detail."
    )
    SUMMARY = (
        "One-line summary: An introductory lecture on deep learning covering gradient descent, "
        "backpropagation, and neural network design.\n\n"
        "Key points:\n"
        "- Gradient descent iteratively minimises the loss function\n"
        "- Backpropagation computes gradients through the chain rule\n"
        "- Neural networks are composed of stacked learnable layers\n\n"
        "Detailed summary: This lecture introduces the core ideas behind deep learning. "
        "Gradient descent is the primary optimisation algorithm used to train models. "
        "Backpropagation enables efficient gradient computation across all layers. "
        "Neural networks learn hierarchical representations from raw data. "
        "These three concepts together form the bedrock of modern AI systems."
    )

    async def fake_extract_audio(data, source):
        return ExtractedInput(
            source=source, type="audio", text=TRANSCRIPT,
            meta={"duration_seconds": 300.0, "chars": len(TRANSCRIPT)},
        )

    async def fake_plan_json(messages, **kw):
        return _plan("summarize")

    async def fake_chat(messages, **kw):
        return SUMMARY

    monkeypatch.setattr(ir, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "Summarize this lecture", "conversation_id": "tc1"},
        files=[("files", ("lecture.wav", _wav_bytes(), "audio/wav"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    assert "One-line summary" in body["final_answer"]
    assert "Key points" in body["final_answer"]
    assert "Detailed summary" in body["final_answer"]
    assert body["final_answer"].count("\n-") >= 3

    assert body["plan_trace"][0]["tool"] == "summarize"

    audio_input = next(i for i in body["extracted_inputs"] if i["type"] == "audio")
    assert audio_input["meta"]["duration_seconds"] == 300.0


def test_tc2_pdf_natural_language_query(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client

    MEETING_NOTES = (
        "Project Sync - 12 June 2025\n"
        "Attendees: Alice, Bob, Carol\n\n"
        "We reviewed Q2 progress and identified the following action items:\n"
        "1. Alice to finalise the API design doc by Friday.\n"
        "2. Bob to set up the staging environment this week.\n"
        "3. Carol to send the updated budget forecast to management.\n\n"
        "Next meeting: 19 June 2025."
    )
    ACTION_ITEMS_ANSWER = (
        "The action items from the meeting are:\n"
        "1. Alice — finalise the API design doc by Friday\n"
        "2. Bob — set up the staging environment this week\n"
        "3. Carol — send the updated budget forecast to management"
    )

    async def fake_plan_json(messages, **kw):
        return _plan("qa")

    async def fake_chat(messages, **kw):
        return ACTION_ITEMS_ANSWER

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "What are the action items?", "conversation_id": "tc2"},
        files=[("files", ("meeting.pdf", _pdf_bytes(MEETING_NOTES), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    answer = body["final_answer"]
    assert "action item" in answer.lower()
    assert "Alice" in answer
    assert "Bob" in answer
    assert "Carol" in answer

    assert body["plan_trace"][0]["tool"] == "qa"

    pdf_input = next(i for i in body["extracted_inputs"] if i["type"] == "pdf")
    assert "alice" in pdf_input["text"].lower()


def test_tc3_image_with_code(monkeypatch):
    from app.agent import planner
    from app.ingestion import image_extractor as ie
    from app.llm.client import llm_client

    CODE_TEXT = (
        "def bubble_sort(arr):\n"
        "    for i in range(len(arr)):\n"
        "        for j in range(len(arr) - i - 1):\n"
        "            if arr[j] > arr[j + 1]:\n"
        "                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n"
        "    return arr"
    )
    EXPLANATION = (
        "What it does: Implements bubble sort — repeatedly swaps adjacent elements "
        "that are out of order until the list is fully sorted.\n\n"
        "Bugs / issues detected:\n"
        "- No early-exit flag: the algorithm always runs O(n^2) comparisons even on "
        "an already-sorted list, which is avoidable\n\n"
        "Time complexity: O(n^2) — two nested loops each iterate up to n times "
        "in the worst case"
    )

    def fake_ocr_image(image):
        return CODE_TEXT, 88.5

    async def fake_plan_json(messages, **kw):
        return _plan("code_explain")

    async def fake_chat(messages, **kw):
        return EXPLANATION

    monkeypatch.setattr(ie, "ocr_image", fake_ocr_image)
    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "Explain", "conversation_id": "tc3"},
        files=[("files", ("code_screenshot.png", _png_bytes(), "image/png"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    answer = body["final_answer"]
    assert "What it does" in answer
    assert "Bugs" in answer
    assert "Time complexity" in answer

    assert body["plan_trace"][0]["tool"] == "code_explain"

    image_input = next(i for i in body["extracted_inputs"] if i["type"] == "image")
    assert image_input["meta"]["ocr_confidence"] == 88.5
    assert "bubble_sort" in image_input["text"]


def test_tc4_pdf_youtube_url_chain(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client
    from app.tools import youtube as yt

    PDF_TEXT = (
        "Recording from the last session:\n"
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
        "Please review before the next meeting."
    )
    YT_TRANSCRIPT = (
        "Welcome. In this video we explore supervised learning and model evaluation. "
        "We cover loss functions, training loops, and how to measure generalisation."
    )
    SUMMARY = (
        "One-line summary: An introduction to supervised learning, loss functions, and model evaluation.\n\n"
        "Key points:\n"
        "- Supervised learning maps inputs to outputs using labelled data\n"
        "- Loss functions measure the gap between predictions and ground truth\n"
        "- Evaluation on held-out data estimates real-world performance\n\n"
        "Detailed summary: This video introduces core supervised learning concepts. "
        "A loss function quantifies how wrong the model's predictions are. "
        "The training loop iteratively adjusts weights to minimise that loss. "
        "Held-out validation and test sets measure generalisation beyond training data. "
        "These ideas underpin almost every practical machine learning system today."
    )

    async def fake_plan_json(messages, **kw):
        return _plan("youtube_transcript", "summarize")

    async def fake_fetch_transcript(video_id):
        return YT_TRANSCRIPT

    async def fake_chat(messages, **kw):
        return SUMMARY

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(yt, "fetch_transcript", fake_fetch_transcript)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={
            "message": "Hit the YT URL in this PDF and give me a summary of it",
            "conversation_id": "tc4",
        },
        files=[("files", ("report.pdf", _pdf_bytes(PDF_TEXT), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    tools_used = [s["tool"] for s in body["plan_trace"]]
    assert tools_used == ["youtube_transcript", "summarize"]
    assert len(body["plan_trace"]) == 2
    assert body["clarification"] is None

    answer = body["final_answer"]
    assert "One-line summary" in answer
    assert "Key points" in answer
    assert "Detailed summary" in answer
    assert answer.count("\n-") >= 3


def test_tc5_multi_file_unified_query(monkeypatch):
    from app.agent import planner
    from app.ingestion import router as ir
    from app.llm.client import llm_client

    AUDIO_TEXT = (
        "In Q2 our revenue grew by twelve percent year over year. "
        "Customer satisfaction scores reached a record high of seventy-two."
    )
    PDF_TEXT = (
        "Q2 Financial Report\n\n"
        "Revenue: up 12% year-over-year\n"
        "Customer NPS: 72 (record high)\n"
        "Operating margin improved by 2 percentage points."
    )
    COMPARISON = (
        "Both the audio recording and the PDF report cover the same topic: Q2 financial results.\n\n"
        "Agreement: Both sources confirm a 12% revenue increase and a record customer satisfaction "
        "score of 72. The data is consistent across modalities.\n\n"
        "Difference: The audio is a spoken summary while the PDF is a formal report that also "
        "mentions the operating margin improvement, which the audio does not cover."
    )

    async def fake_extract_audio(data, source):
        return ExtractedInput(
            source=source, type="audio", text=AUDIO_TEXT,
            meta={"duration_seconds": 45.0, "chars": len(AUDIO_TEXT)},
        )

    async def fake_plan_json(messages, **kw):
        return _plan("compare")

    async def fake_chat(messages, **kw):
        return COMPARISON

    monkeypatch.setattr(ir, "extract_audio", fake_extract_audio)
    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={
            "message": "Do the audio and the document discuss the same topic?",
            "conversation_id": "tc5",
        },
        files=[
            ("files", ("recording.wav", _wav_bytes(), "audio/wav")),
            ("files", ("report.pdf", _pdf_bytes(PDF_TEXT), "application/pdf")),
        ],
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["plan_trace"][0]["tool"] == "compare"
    assert body["clarification"] is None

    answer = body["final_answer"]
    assert "same topic" in answer.lower() or "revenue" in answer.lower()
    assert "12" in answer

    types = {inp["type"] for inp in body["extracted_inputs"]}
    assert "audio" in types
    assert "pdf" in types
