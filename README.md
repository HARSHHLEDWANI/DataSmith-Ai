# Agentic Multi-Modal AI

A FastAPI application that accepts text, images, PDFs and audio in a single request, figures out what you want across all of them, and plans and runs a tool chain to get it done. If the goal is ambiguous it asks one follow-up question instead of guessing. Every output is plain text.

Built for a 48-hour assignment. Single deployable service — no separate frontend build, no database.

## Features

- Multi-file upload in one request: images (JPG/PNG), PDFs (text or scanned), audio (MP3/WAV/M4A), plain text
- OCR for images and scanned PDF pages, with per-page confidence scores
- Audio transcription via Groq's hosted Whisper API, with duration
- YouTube URLs detected anywhere in any input (including inside a PDF) and fetched automatically
- LLM-generated plans: the agent decides which tools to run and in what order per request
- Clarification gate: if the goal is missing or ambiguous, it asks before doing anything
- Tool chaining: steps pipe their output forward (e.g. fetch transcript → summarize it)
- Streaming final answer via SSE, live step-by-step trace in the UI, cost estimate
- Provider-agnostic LLM: Groq by default, swap to DeepSeek or OpenAI via env vars

## The 8 tasks

| Task | How |
|------|-----|
| Image/PDF text extraction + OCR confidence | `ingestion/image_extractor.py`, `ingestion/pdf_extractor.py` |
| YouTube transcript (URL anywhere in any input) | `tools/youtube.py` |
| Conversational Q&A | `tools/qa.py` |
| Summarization (1-line + 3 bullets + 5-sentence paragraph, always) | `tools/summarize.py` |
| Sentiment (label + confidence + justification) | `tools/sentiment.py` |
| Code explanation (what it does + bugs + time complexity) | `tools/code_explain.py` |
| Audio transcription + summary + duration | `ingestion/audio_extractor.py` |
| Cross-input reasoning (compare/combine multiple inputs) | `tools/compare.py` |

## Project layout

```
app/
  main.py                 # FastAPI routes, static serving, SSE, CORS
  config.py               # all env vars via pydantic-settings
  logging_config.py       # JSON structured logging
  agent/
    orchestrator.py       # plan → clarify? → cost → execute → respond
    planner.py            # LLM call → strict JSON plan
    executor.py           # runs steps, chains outputs, retries, builds trace
    cost.py               # pre-execution token/cost estimate
    session_store.py      # in-memory pending-clarification + run stores
    trace.py              # Plan/StepResult/RunState dataclasses
  ingestion/
    router.py             # type detection + dispatch
    image_extractor.py    # OCR + confidence
    pdf_extractor.py      # PyMuPDF + per-page OCR fallback
    audio_extractor.py    # Groq Whisper + duration
    text_extractor.py     # decode + normalize
    models.py             # ExtractedInput
  tools/
    registry.py           # tool registry
    summarize.py
    sentiment.py
    code_explain.py
    youtube.py            # URL detection across all inputs + transcript fetch
    qa.py
    compare.py
  llm/
    client.py             # provider-agnostic chat/JSON/stream + retries
  static/                 # index.html, app.js, style.css
tests/                    # ingestion, tools, planner, e2e + generated fixtures
```

## Local setup (no Docker)

Requires Python 3.11+ and Tesseract OCR (for image/scanned-PDF tasks).

Install Tesseract:
- macOS: `brew install tesseract`
- Ubuntu: `sudo apt-get install tesseract-ocr`
- Windows: [UB-Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki), add to PATH

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env             # then fill in your keys
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000. API docs at http://localhost:8000/docs.

The app starts without any keys — the UI and `/docs` load fine, features that need a key return a clear message instead of crashing.

## Local setup (Docker)

```bash
cp .env.example .env    # fill in keys
docker compose up --build
```

Tesseract and ffmpeg are included in the image.

## Environment variables

| Variable | Default | What it does |
|----------|---------|--------------|
| `LLM_API_KEY` | *(empty)* | API key for chat completions — required to run the agent |
| `GROQ_API_KEY` | *(empty)* | Key for Whisper audio transcription (can be same as above if using Groq) |
| `LLM_PROVIDER` | `groq` | Informational label |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` | OpenAI-compatible base URL |
| `LLM_MODEL` | `llama-3.3-70b-versatile` | Chat model |
| `MAX_FILE_SIZE_MB` | `25` | Per-file upload limit |
| `LLM_TIMEOUT_S` | `60` | Per-call timeout |
| `MAX_CONTEXT_CHARS` | `6000` | Per-input truncation budget for the planner |
| `PORT` | `8000` | Server port (Render sets this automatically) |

Get a free Groq key at https://console.groq.com. One key covers both chat and Whisper.

Switching to DeepSeek:
```
LLM_PROVIDER=deepseek
LLM_API_KEY=your_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL=deepseek-chat
```

## Deploying to Render

1. Push this repo to GitHub.
2. Render dashboard → New → Web Service → connect repo.
3. Runtime: Docker. Instance type: Free. Health check path: `/health`.
4. Add environment variables: `LLM_PROVIDER`, `LLM_BASE_URL`, `LLM_MODEL`, and mark `LLM_API_KEY` + `GROQ_API_KEY` as secrets.
5. Create Web Service. First build takes a few minutes (installs tesseract + ffmpeg).

Alternatively use the included `render.yaml` via New → Blueprint.

One thing to know about the free tier: services sleep after about 15 minutes idle. The first request after waking takes 30-50 seconds. Conversation state is in-memory so it resets on restart — any pending clarification questions are cleared.

## Sample test cases

**1. PDF or image extraction**
Upload a PDF or screenshot, type "Summarize this." The agent OCR-extracts the text (with confidence score for scanned pages) and runs the summarizer: one-line summary, three bullets, five-sentence paragraph.

**2. YouTube link inside a PDF**
Upload a PDF that has a YouTube URL in it, type "Fetch the YouTube link in this PDF and summarize the video." The planner chains youtube_transcript → summarize with no extra steps from you. If captions are disabled on the video, it returns a message saying so instead of an error.

**3. Audio transcription**
Upload an MP3 or WAV, type "Transcribe and summarize this." Whisper transcribes it (audio duration shown in the extracted-text panel), then the summarizer runs on the transcript.

**4. Code explanation**
Paste or upload code with "Explain this code, find bugs, and give time complexity." Returns what it does, any bugs found, and a Big-O analysis.

**5. Cross-input reasoning + ambiguity gate**
Upload two documents and type "Do these discuss the same topic?" — the compare tool reasons across both. Or upload a file with no instruction at all — the agent asks what you want to do with it before touching any tools.

## API

FastAPI serves interactive docs at `/docs` and OpenAPI JSON at `/openapi.json`.

Key endpoints:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Chat UI |
| GET | `/health` | Liveness + which features are configured |
| GET | `/tools` | Tool registry |
| POST | `/chat` | Agent turn. Multipart: `message`, `conversation_id`, `files[]`. Returns `extracted_inputs`, `plan_trace`, `cost`, `final_answer`, `clarification`, `run_id` |
| GET | `/runs/{id}/status` | Poll run progress |
| GET | `/runs/{id}/stream` | SSE stream of the final answer |

```bash
curl -s -X POST http://localhost:8000/chat \
  -F "message=Summarize this" \
  -F "files=@report.pdf" | python -m json.tool
```

## Design decisions

**Why FastAPI over Flask/Django.** Native async matters because ingesting multiple files and calling the LLM happen concurrently. Pydantic validation and auto-generated `/docs` were useful enough to be worth the choice. Django would have been overkill.

**Planner + executor instead of one big prompt.** A single prompt that does everything is opaque and fragile. Splitting "what to do" (LLM produces a JSON plan) from "how to do it" (Python runs the steps) means every tool call is recorded, retried independently, and surfaced in the trace. The plan also caps at 6 steps so there's no runaway loop.

**Tool registry.** Capabilities are data rather than branches. The planner reads tool descriptions to choose; adding a new tool is one file, and the planner can use it immediately. It also structurally enforces text-only output — every tool callable returns `str`.

**Hosted Whisper instead of local.** I used Groq's Whisper API instead of running whisper locally because Render's free tier doesn't have the memory or disk for the model weights. The container stays under ~500MB this way.

**PyMuPDF over pdfplumber/pypdf.** It's faster and — critically — can render a page to an image, which is exactly what the scanned-PDF OCR fallback needs. pdfplumber and pypdf can't do that.

**Provider-agnostic LLM client.** Groq, DeepSeek and OpenAI all speak the OpenAI wire format. One SDK client with a configurable `base_url` means switching providers is three env vars.

**In-memory state.** Session and run state live in process memory. It's fine for the assignment and keeps the deploy simple. The trade-off is that state resets on restart and isn't shared across workers. Redis would slot in behind the same repository interface if this needed to scale.

**Vanilla JS frontend.** No build step, no Node in the container, nothing to break. The UI is simple enough that a framework would add more weight than value.
