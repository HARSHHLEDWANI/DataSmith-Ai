from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.agent.orchestrator import PlanNotFoundError, PreparedPlan, orchestrator
from app.agent.plan_ops import validate_edited_steps
from app.agent.session_store import plan_store, run_store
from app.agent.trace import RunState
from app.config import settings
from app.ingestion.models import ExtractedInput
from app.ingestion.router import ingest_file
from app.ingestion.text_extractor import extract_plain
from app.llm.client import LLMError, llm_client
from app.logging_config import get_logger, setup_logging
from app.tools.registry import register_all_tools

setup_logging(settings.log_level)
logger = get_logger("main")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="triage",
    description=(
        "Agent for messy, composite input: accepts PDFs (including links buried "
        "inside them), images, audio, and questions in one request, plans a "
        "visible tool chain, and answers with source attribution."
    ),
    version="1.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

register_all_tools()

_NO_CACHE = {"Cache-Control": "no-cache"}


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers=_NO_CACHE)


@app.get("/static/{filename}", include_in_schema=False)
async def static_files(filename: str) -> FileResponse:
    safe = Path(filename).name
    path = STATIC_DIR / safe
    if not path.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, headers=_NO_CACHE)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "status": "ok",
        "llm_configured": settings.llm_configured(),
        "whisper_configured": settings.whisper_configured(),
        "provider": settings.llm_provider,
        "model": settings.llm_model,
    }


@app.get("/tools")
async def list_tools() -> dict[str, object]:
    return {"tools": register_all_tools().as_list()}


async def _ingest_uploads(files: list[UploadFile]) -> list[ExtractedInput]:
    async def _one(f: UploadFile) -> ExtractedInput:
        data = await f.read()
        return await ingest_file(data, f.filename or "upload", f.content_type)

    if not files:
        return []
    return list(await asyncio.gather(*[_one(f) for f in files]))


@app.post("/chat")
async def chat(
    message: str = Form(default=""),
    conversation_id: str = Form(default=""),
    files: list[UploadFile] | None = None,
) -> JSONResponse:
    conversation_id = conversation_id or uuid.uuid4().hex
    run_id = uuid.uuid4().hex

    inputs = await _ingest_uploads(files or [])
    if message.strip():
        inputs.append(extract_plain(message, source="user_query"))

    run_store.create(run_id)
    try:
        state = await orchestrator.run(
            conversation_id=conversation_id,
            query=message,
            inputs=inputs,
            run_id=run_id,
        )
    except LLMError as exc:
        return JSONResponse(
            status_code=200,
            content={
                "conversation_id": conversation_id,
                "run_id": run_id,
                "error": str(exc),
                "extracted_inputs": [i.model_dump() for i in inputs],
                "plan_trace": [],
                "final_answer": f"Error: {exc}",
                "clarification": None,
            },
        )

    return JSONResponse(content=_serialize_state(state, conversation_id))


def _serialize_state(state: RunState, conversation_id: str) -> dict[str, object]:
    return {
        "conversation_id": conversation_id,
        "run_id": state.run_id,
        "status": state.status,
        "extracted_inputs": state.extracted_inputs,
        "detected_inputs": state.detected_references,
        "input_manifest": state.input_manifest,
        "plan_trace": [s.model_dump() for s in state.steps],
        "cost": state.cost.model_dump() if state.cost else None,
        "final_answer": state.final_answer,
        "clarification": state.clarifying_question,
        "error": state.error,
    }


def _serialize_prepared(p: PreparedPlan) -> dict[str, object]:
    return {
        "conversation_id": p.conversation_id,
        "plan_id": p.plan_id,
        "needs_clarification": p.clarifying_question is not None,
        "clarifying_question": p.clarifying_question,
        "steps": p.steps,
        "detected_inputs": p.detected_refs,
        "input_manifest": p.manifest,
        "cost": p.cost.model_dump() if p.cost else None,
        "available_tools": register_all_tools().as_list(),
    }


@app.post("/plan")
async def plan_endpoint(
    message: str = Form(default=""),
    conversation_id: str = Form(default=""),
    replan_notes: str = Form(default=""),
    files: list[UploadFile] | None = None,
) -> JSONResponse:
    """Stage 1: ingest + plan, but do NOT execute. Returns an editable plan."""
    conversation_id = conversation_id or uuid.uuid4().hex
    inputs = await _ingest_uploads(files or [])
    if message.strip():
        inputs.append(extract_plain(message, source="user_query"))

    try:
        prepared = await orchestrator.prepare_plan(
            conversation_id=conversation_id,
            query=message,
            inputs=inputs,
            replan_notes=replan_notes or None,
        )
    except LLMError as exc:
        return JSONResponse(
            status_code=200,
            content={
                "conversation_id": conversation_id,
                "plan_id": None,
                "error": str(exc),
                "steps": [],
                "needs_clarification": False,
                "extracted_inputs": [i.model_dump() for i in inputs],
            },
        )

    body = _serialize_prepared(prepared)
    body["extracted_inputs"] = [i.model_dump() for i in inputs]
    return JSONResponse(content=body)


class ExecuteRequest(BaseModel):
    plan_id: str
    conversation_id: str = ""
    steps: list[dict] = []


@app.post("/execute")
async def execute_endpoint(req: ExecuteRequest) -> JSONResponse:
    """Stage 2a: validate the (possibly edited) plan and mint a run_id. Actual
    execution happens when the client opens the execute_stream for that run."""
    staged = plan_store.get(req.plan_id)
    if staged is None:
        raise HTTPException(status_code=404, detail="Unknown or expired plan_id — re-plan first.")

    validated, warnings = validate_edited_steps(req.steps, register_all_tools())
    staged.plan = validated
    run_id = uuid.uuid4().hex
    run_store.create(run_id)
    return JSONResponse(
        content={"run_id": run_id, "plan_id": req.plan_id, "warnings": warnings, "executable": True}
    )


def _step_view(step) -> dict[str, object]:
    return {
        "id": step.step,
        "tool": step.tool,
        "status": step.status,
        "description": step.reasoning,
        "duration_ms": step.duration_ms,
        "output_preview": (step.output or "")[:240],
        "error": step.error,
    }


def _progress_event(state: RunState) -> dict[str, object]:
    return {
        "type": "progress",
        "current_step": state.current_step,
        "steps": [_step_view(s) for s in state.steps],
    }


@app.get("/runs/{run_id}/execute_stream")
async def execute_stream(run_id: str, plan_id: str) -> StreamingResponse:
    """Stage 2b: run the staged plan, emitting live per-step SSE events. Execution
    happens inside this request (no background worker); disconnect cancels it."""
    staged = plan_store.get(plan_id)
    if staged is None or staged.plan is None:
        raise HTTPException(status_code=404, detail="Unknown or expired plan_id — re-plan first.")
    plan = staged.plan

    async def event_gen():
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        async def progress(state: RunState) -> None:
            await queue.put(_progress_event(state))

        async def drive() -> None:
            try:
                state = await orchestrator.execute_staged(
                    run_id=run_id, plan_id=plan_id, plan=plan, progress=progress
                )
                await queue.put(
                    {
                        "type": "answer",
                        "final_answer": state.final_answer,
                        "any_failed": any(s.status == "failure" for s in state.steps),
                    }
                )
            except (PlanNotFoundError, LLMError) as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception as exc:  # never leave the stream hanging
                logger.warning("execute_stream failed", extra={"data": {"error": str(exc)}})
                await queue.put({"type": "error", "message": str(exc)})
            finally:
                await queue.put(sentinel)

        task = asyncio.create_task(drive())
        yield f"data: {json.dumps({'type': 'run_started', 'run_id': run_id, 'total_steps': len(plan.plan)})}\n\n"
        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                yield f"data: {json.dumps(item)}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        finally:
            if not task.done():
                task.cancel()

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}/status")
async def run_status(run_id: str) -> JSONResponse:
    state = run_store.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown run id")
    return JSONResponse(content=_serialize_state(state, ""))


@app.get("/runs/{run_id}/stream")
async def run_stream(run_id: str) -> StreamingResponse:
    state = run_store.get(run_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Unknown run id")

    async def event_gen():
        answer = state.final_answer
        if answer:
            for i in range(0, len(answer), 24):
                yield f"data: {json.dumps({'token': answer[i:i+24]})}\n\n"
                await asyncio.sleep(0.02)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/runs/{run_id}/stream_live")
async def run_stream_live(run_id: str, q: str = "") -> StreamingResponse:
    async def event_gen():
        try:
            async for token in llm_client.stream(
                [{"role": "user", "content": q or "Say hello."}], max_tokens=400
            ):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except LLMError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
