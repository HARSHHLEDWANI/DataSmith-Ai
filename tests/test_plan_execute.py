"""Tests for the two-stage plan/execute flow, plan validation, and live SSE."""
from __future__ import annotations

import json

import fitz
from fastapi.testclient import TestClient


def _pdf_bytes(text: str) -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_textbox(fitz.Rect(50, 50, 545, 742), text, fontsize=11)
    data = doc.tobytes()
    doc.close()
    return data


def _app_client() -> TestClient:
    from app import main
    return TestClient(main.app)


def _linear_plan(*tools_and_deps):
    plan = []
    for i, (tool, dep) in enumerate(tools_and_deps, start=1):
        plan.append({"step": i, "tool": tool, "input_from": dep, "reasoning": f"{tool} step"})
    return {"needs_clarification": False, "clarifying_question": None, "plan": plan}


# ---- plan_ops.validate_edited_steps (unit) ---------------------------------

def test_validate_drops_unknown_tool():
    from app.agent.plan_ops import validate_edited_steps
    from app.tools.registry import register_all_tools

    reg = register_all_tools()
    steps = [
        {"id": 1, "tool": "summarize", "input_from": "context"},
        {"id": 2, "tool": "not_a_tool", "input_from": "context"},
    ]
    plan, warnings = validate_edited_steps(steps, reg)
    assert [s.tool for s in plan.plan] == ["summarize"]
    assert any("unknown tool" in w for w in warnings)


def test_validate_degrades_removed_dependency():
    from app.agent.plan_ops import validate_edited_steps
    from app.tools.registry import register_all_tools

    reg = register_all_tools()
    # Keep only the step that depended on the removed step 1.
    steps = [{"id": 2, "tool": "summarize", "input_from": "step:1"}]
    plan, warnings = validate_edited_steps(steps, reg)
    assert plan.plan[0].input_from == "context"
    assert any("removed" in w for w in warnings)


def test_validate_reorder_remaps_dependency():
    from app.agent.plan_ops import validate_edited_steps
    from app.tools.registry import register_all_tools

    reg = register_all_tools()
    # Original: step1 youtube_transcript, step2 summarize(step:1).
    # Submitted in the SAME order keeps the dependency valid and remapped.
    steps = [
        {"id": 1, "tool": "youtube_transcript", "input_from": "context"},
        {"id": 2, "tool": "summarize", "input_from": "step:1"},
    ]
    plan, warnings = validate_edited_steps(steps, reg)
    assert plan.plan[1].input_from == "step:1"
    assert warnings == []

    # Now reverse the order: summarize moved before its dependency -> degrade + warn.
    reversed_steps = [
        {"id": 2, "tool": "summarize", "input_from": "step:1"},
        {"id": 1, "tool": "youtube_transcript", "input_from": "context"},
    ]
    plan2, warnings2 = validate_edited_steps(reversed_steps, reg)
    assert plan2.plan[0].tool == "summarize"
    assert plan2.plan[0].input_from == "context"
    assert any("moved before" in w for w in warnings2)


def test_validate_compound_dependency_resolves_without_warning():
    from app.agent.plan_ops import validate_edited_steps
    from app.tools.registry import register_all_tools

    reg = register_all_tools()
    # The planner sometimes emits a compound ref like "step:2, step:3".
    # Unedited, it should resolve to the first valid earlier step, no warning.
    steps = [
        {"id": 1, "tool": "youtube_transcript", "input_from": "context"},
        {"id": 2, "tool": "summarize", "input_from": "step:1"},
        {"id": 3, "tool": "summarize", "input_from": "context"},
        {"id": 4, "tool": "compare", "input_from": "step:2, step:3"},
    ]
    plan, warnings = validate_edited_steps(steps, reg)
    assert plan.plan[3].input_from == "step:2"
    assert warnings == []


def test_validate_empty_falls_back_to_qa():
    from app.agent.plan_ops import validate_edited_steps
    from app.tools.registry import register_all_tools

    plan, warnings = validate_edited_steps([], register_all_tools())
    assert [s.tool for s in plan.plan] == ["qa"]
    assert warnings


# ---- POST /plan -------------------------------------------------------------

def test_plan_returns_editable_schema_without_executing(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client

    chat_calls = []

    async def fake_plan_json(messages, **kw):
        return _linear_plan(("youtube_transcript", "context"), ("summarize", "step:1"))

    async def fake_chat(messages, **kw):
        chat_calls.append(1)
        return "should not be called during planning"

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/plan",
        data={"message": "Summarize the linked video", "conversation_id": "plan1"},
        files=[("files", ("report.pdf", _pdf_bytes("Watch https://youtu.be/dQw4w9WgXcQ now."), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()

    assert body["plan_id"]
    assert [s["id"] for s in body["steps"]] == [1, 2]
    assert [s["tool"] for s in body["steps"]] == ["youtube_transcript", "summarize"]
    assert body["steps"][0]["instructions"] == ""
    assert {t["name"] for t in body["available_tools"]} >= {"qa", "summarize", "youtube_transcript"}
    assert body["detected_inputs"][0]["kind"] == "youtube"
    assert not chat_calls  # no tools ran during planning


def test_plan_clarification(monkeypatch):
    from app.agent import planner

    async def fake_plan_json(messages, **kw):
        return {
            "needs_clarification": True,
            "clarifying_question": "What would you like me to do with this file?",
            "plan": [],
        }

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)

    resp = _app_client().post(
        "/plan",
        data={"message": "", "conversation_id": "plan-clar"},
        files=[("files", ("report.pdf", _pdf_bytes("Some content."), "application/pdf"))],
    )
    body = resp.json()
    assert body["needs_clarification"] is True
    assert body["clarifying_question"]
    assert body["steps"] == []
    assert body["plan_id"] is None


# ---- POST /execute + GET execute_stream ------------------------------------

def _consume_sse(client, run_id, plan_id):
    events = []
    with client.stream("GET", f"/runs/{run_id}/execute_stream?plan_id={plan_id}") as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line and line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_execute_stream_runs_and_applies_instructions(monkeypatch):
    from app.agent import planner
    from app.llm.client import llm_client
    from app.tools import youtube as yt

    captured = []

    async def fake_plan_json(messages, **kw):
        return _linear_plan(("youtube_transcript", "context"), ("summarize", "step:1"))

    async def fake_fetch(video_id):
        return "transcript body"

    async def fake_chat(messages, **kw):
        captured.append(messages[-1]["content"])
        return "THE SUMMARY"

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(yt, "fetch_transcript", fake_fetch)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    client = _app_client()
    plan = client.post(
        "/plan",
        data={"message": "Summarize the linked video", "conversation_id": "exec1"},
        files=[("files", ("report.pdf", _pdf_bytes("Watch https://youtu.be/dQw4w9WgXcQ now."), "application/pdf"))],
    ).json()

    steps = plan["steps"]
    steps[1]["instructions"] = "Answer in exactly one sentence."

    ex = client.post("/execute", json={"plan_id": plan["plan_id"], "conversation_id": "exec1", "steps": steps})
    assert ex.status_code == 200
    assert ex.json()["warnings"] == []
    run_id = ex.json()["run_id"]

    events = _consume_sse(client, run_id, plan["plan_id"])
    types = [e["type"] for e in events]
    assert types[0] == "run_started"
    assert "progress" in types
    assert types[-1] == "done"

    answer = next(e for e in events if e["type"] == "answer")
    assert answer["final_answer"] == "THE SUMMARY"
    assert answer["any_failed"] is False

    # the summarize step's per-step instruction reached the tool prompt
    assert any("exactly one sentence" in c for c in captured)

    # steps progressed to success in the final progress snapshot
    last_progress = [e for e in events if e["type"] == "progress"][-1]
    assert all(s["status"] == "success" for s in last_progress["steps"])


def test_execute_unknown_plan_id_404():
    client = _app_client()
    resp = client.post("/execute", json={"plan_id": "does-not-exist", "steps": []})
    assert resp.status_code == 404


def test_chat_backward_compatible(monkeypatch):
    """The one-shot /chat path is untouched and still returns its full schema."""
    from app.agent import planner
    from app.llm.client import llm_client

    async def fake_plan_json(messages, **kw):
        return _linear_plan(("qa", "context"))

    async def fake_chat(messages, **kw):
        return "Alice owns the API doc."

    monkeypatch.setattr(planner.llm_client, "chat_json", fake_plan_json)
    monkeypatch.setattr(llm_client, "chat", fake_chat)

    resp = _app_client().post(
        "/chat",
        data={"message": "Who owns the API doc?", "conversation_id": "compat1"},
        files=[("files", ("notes.pdf", _pdf_bytes("Alice will finalise the API design doc."), "application/pdf"))],
    )
    assert resp.status_code == 200
    body = resp.json()
    for key in ("plan_trace", "final_answer", "detected_inputs", "input_manifest", "extracted_inputs"):
        assert key in body
    assert body["plan_trace"][0]["tool"] == "qa"
