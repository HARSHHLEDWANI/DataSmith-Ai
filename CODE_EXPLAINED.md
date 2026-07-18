# triage — Complete Code Walkthrough

This document explains **every code file** in the project, what it does, and how the pieces
connect. It's organized to follow the path a request actually takes:

> **upload → ingestion → agent (plan + execute) → tools → LLM → response**

Each section names the file, says what its job is, then walks through the important code.
Read it top to bottom and you'll understand the whole program.

---

## Table of contents

1. [Config & infrastructure](#1-config--infrastructure) — `config.py`, `logging_config.py`
2. [Data models](#2-data-models) — `ingestion/models.py`, `agent/trace.py`
3. [Ingestion: files → text](#3-ingestion-files--text) — `router.py`, `text_extractor.py`, `image_extractor.py`, `pdf_extractor.py`, `audio_extractor.py`
4. [The LLM client](#4-the-llm-client) — `llm/client.py`
5. [The tool system](#5-the-tool-system) — `tools/registry.py` + the 6 tools
6. [The agent brain](#6-the-agent-brain) — `planner.py`, `executor.py`, `orchestrator.py`, `cost.py`, `session_store.py`
7. [The web layer](#7-the-web-layer) — `main.py`, `static/`
8. [The tests](#8-the-tests) — `conftest.py`, `test_sample_cases.py`
9. [How one request flows end-to-end](#9-how-one-request-flows-end-to-end)

---

## 1. Config & infrastructure

### [app/config.py](app/config.py) — all settings in one place

Uses `pydantic-settings` so every setting can come from an environment variable or the `.env`
file. The `Settings` class declares typed fields with defaults:

- **LLM settings** — `llm_provider` (default `groq`), `llm_api_key`, `llm_base_url`
  (`https://api.groq.com/openai/v1`), `llm_model` (`llama-3.3-70b-versatile`). Because the base
  URL is configurable, the same code works with Groq, OpenAI, or DeepSeek.
- **Whisper (audio) settings** — `groq_api_key`, `groq_base_url`, `whisper_model`
  (`whisper-large-v3`).
- **Limits / budgets** — `max_file_size_mb` (25), various timeouts, and character budgets
  (`max_context_chars`, `max_tool_input_chars`, `max_conversation_chars`) that keep prompts from
  growing without bound.

Helper methods make intent obvious elsewhere in the code:
```python
def llm_configured(self) -> bool:        # is an LLM key set?
    return bool(self.llm_api_key)

def whisper_configured(self) -> bool:    # can we transcribe audio?
    return bool(self.groq_api_key or (self.llm_api_key and "groq" in self.llm_base_url))

def whisper_key(self) -> str:            # which key to use for Whisper
    return self.groq_api_key or self.llm_api_key
```
`get_settings()` is wrapped in `@lru_cache` so settings are read once and reused. The module ends
with `settings = get_settings()` — a single shared object imported everywhere.

### [app/logging_config.py](app/logging_config.py) — structured JSON logs

`JsonFormatter` turns every log record into a single JSON line (timestamp, level, logger name,
message). The clever bit: if a log call passes `extra={"data": {...}}`, those fields are merged
into the JSON. That's why you see calls like `logger.info("ingesting file", extra={"data": {...}})`
all over the codebase — they produce machine-readable, searchable logs. `setup_logging()` wires
this formatter to stdout; `get_logger(name)` is the accessor every module uses.

---

## 2. Data models

These two files define the "shapes" of data that flow through the program. Everything else passes
these objects around.

### [app/ingestion/models.py](app/ingestion/models.py) — `ExtractedInput`

This is **the single most important data type in the project**. Every uploaded file *and* the
user's typed message becomes one `ExtractedInput`:

```python
class ExtractedInput(BaseModel):
    source: str          # filename or label, e.g. "report.pdf" or "user_query"
    type: str            # "text" | "image" | "pdf" | "audio" | "unknown"
    text: str = ""       # the extracted / transcribed text
    meta: dict = {}      # extras: ocr_confidence, duration_seconds, pages, chars...
    error: str | None    # set if extraction failed (file still flows through, just empty)

    @property
    def ok(self) -> bool:
        return self.error is None
```
The key design idea: **a failed extraction is not an exception** — it's an `ExtractedInput` with
`error` set and `ok == False`. The rest of the app can keep running and simply skip inputs where
`ok` is false. This is why one bad file never crashes a whole request.

### [app/agent/trace.py](app/agent/trace.py) — the plan & run state models

These Pydantic models are what the agent reasons over and what the API returns:

- **`PlanStep`** — one step the LLM wants to run: `step` number, `tool` name, `input_from`
  (`"context"` / `"query"` / `"step:N"`), and `reasoning`.
- **`Plan`** — the planner's full output: `needs_clarification` (bool), `clarifying_question`, and
  `plan` (a list of `PlanStep`).
- **`StepResult`** — the *result* of running a step: status (`pending/running/success/failure/
  skipped`), `output`, `duration_ms`, `error`, an `input_summary`.
- **`RunState`** — the whole live state of one request: status, the list of `StepResult`s,
  `final_answer`, `clarifying_question`, `cost`, and `extracted_inputs`. This object is what gets
  serialized into the JSON response.
- **`CostEstimate`** — estimated input/output tokens and USD.
- **`StepTimer`** — a small context manager (`with StepTimer(result):`) that times a step, sets
  its status to `running` on enter, and on exit records `duration_ms` and flips status to
  `success` (or `failure` if an exception escaped). It returns `False` from `__exit__` so the
  exception still propagates — letting the executor handle retry/degrade.

---

## 3. Ingestion: files → text

The ingestion layer's only job: take raw bytes and produce an `ExtractedInput`. The agent never
sees a PDF or a PNG — only text.

### [app/ingestion/router.py](app/ingestion/router.py) — the dispatcher

`detect_type(source, content_type)` decides what kind of file this is, first by **file extension**
(sets like `IMAGE_EXTS`, `PDF_EXTS`, `AUDIO_EXTS`, `TEXT_EXTS`), then falling back to the **MIME
type** if the extension is unknown. Returns `"unknown"` if nothing matches.

`ingest_file(data, source, content_type)` is the gatekeeper:
1. Rejects **empty files** and files over the size limit (returns an `ExtractedInput` with `error`).
2. Calls `detect_type`, logs it.
3. Dispatches to the right extractor: `extract_image`, `extract_pdf`, `extract_audio`,
   or `extract_text`. (Note `extract_audio` is `async` because it makes a network call;
   the others are synchronous.)

### [app/ingestion/text_extractor.py](app/ingestion/text_extractor.py) — text + the shared normalizer

- **`normalize_text(raw)`** — used by *every* extractor. Converts `\r\n`→`\n`, strips trailing
  whitespace per line, and collapses runs of blank lines down to a single blank line. This keeps
  prompts clean and compact.
- **`extract_text(data, source)`** — decodes bytes as UTF-8, falling back to latin-1 with
  replacement so it never throws on weird encodings.
- **`extract_plain(text, source="user_query")`** — wraps the user's *typed message* as an
  `ExtractedInput`. This is why the text you type is treated as just another input alongside files.

### [app/ingestion/image_extractor.py](app/ingestion/image_extractor.py) — OCR

Turns an image into text using **Tesseract** (via `pytesseract`):

- **`_ocr_confidence(image)`** — calls `pytesseract.image_to_data(...)` to get per-word text *and*
  confidence scores. It joins the words into a string and averages the valid confidences (ignoring
  Tesseract's `-1` "no confidence" marker) into a single `mean_conf` percentage.
- **`ocr_image(image)`** — pre-processes the image first: `ImageOps.grayscale` then
  `autocontrast` (this measurably improves OCR accuracy on screenshots), then runs the OCR and
  normalizes the text. Returns `(text, confidence)`.
- **`extract_image(data, source)`** — opens the bytes with Pillow (returns an error input if the
  image is unreadable), runs `ocr_image`, and **specifically catches `TesseractNotFoundError`** to
  give a clear "Tesseract OCR engine not installed" message. On success it stores text plus
  `meta` = `ocr_confidence`, `width`, `height`, `chars`.

### [app/ingestion/pdf_extractor.py](app/ingestion/pdf_extractor.py) — PDF text with OCR fallback

Uses **PyMuPDF** (`fitz`). This is the cleverest extractor:

```python
for index, page in enumerate(doc):
    text = (page.get_text() or "").strip()
    if len(text) < MIN_TEXT_CHARS:           # page has little/no embedded text...
        pix = page.get_pixmap(matrix=...)    # ...so render it to an image at 200 DPI
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        ocr_text, conf = ocr_image(image)    # ...and OCR it (reusing the image extractor!)
        if ocr_text.strip():
            text = ocr_text
            ocr_pages.append(page_num)
            ocr_confidences.append(conf)
    page_texts.append(text)
```
So **digital PDFs** are read instantly via `get_text()`, but **scanned PDFs** (image-only pages)
automatically fall back to rendering + OCR, page by page. The `meta` records total `pages`, which
`ocr_pages` were OCR'd, the mean `ocr_confidence`, and `chars`. If nothing extractable is found,
it returns an error input. Note it reuses `ocr_image` from the image extractor — no duplicated OCR
logic.

### [app/ingestion/audio_extractor.py](app/ingestion/audio_extractor.py) — speech-to-text

Turns audio into text using **Groq's Whisper** API:

- **`_duration_seconds(data, source)`** — probes the audio's length with `mutagen` (best-effort;
  returns `None` if it can't, never throws).
- **`_transcribe(data, source)`** — POSTs the audio bytes to Groq's
  `/audio/transcriptions` endpoint with `httpx`, asking for JSON, and returns the `"text"` field.
  It **retries once** on failure with a 1-second pause.
- **`extract_audio(data, source)`** — the public entry. If Whisper isn't configured it returns an
  error input (but still includes the probed duration in `meta`). Otherwise it transcribes,
  normalizes, and records `duration_seconds` + `chars` in `meta`.

---

## 4. The LLM client

### [app/llm/client.py](app/llm/client.py) — talking to the model

A single wrapper around the **async OpenAI SDK**, configured with a custom `base_url` so it speaks
to Groq/DeepSeek/OpenAI interchangeably. Two custom exceptions: `LLMError` (general) and
`LLMNotConfiguredError` (no API key — handled specially so it's never retried).

- **`_ensure_client()`** — lazily builds the `AsyncOpenAI` client; raises
  `LLMNotConfiguredError` with a helpful message if no key is set. `max_retries=0` because *we*
  handle retries.
- **`_with_retry(coro_factory, what)`** — wraps any LLM call: enforces a timeout
  (`asyncio.wait_for`), retries once with backoff, logs failures, and re-raises a clean `LLMError`
  if both attempts fail. Never retries `LLMNotConfiguredError`.
- **`chat(messages, ...)`** — a normal chat completion, returns the text. Used by the
  summarize/qa/compare/etc. tools.
- **`chat_json(messages, ...)`** — used by the **planner**, and the most defensive function in the
  file. It:
  1. Sets `response_format={"type": "json_object"}` to ask the model for JSON.
  2. Guarantees the word "json" appears in the messages (some providers, e.g. Groq, *require* it
     when using JSON mode) — if not present, it prepends a system message.
  3. Parses the result through `_extract_json_object()`, which strips ```` ```json ```` fences and
     **brace-matches** to pull out the first complete `{...}` object even if the model added prose.
  4. **If parsing still fails, it re-prompts the model once** ("your previous response was not
     valid JSON…") and tries again. Only then does it give up with `LLMError`.
- **`stream(messages, ...)`** — async generator yielding tokens as they arrive, used by the live
  streaming endpoint.

This file is why the agent is robust: flaky models, fenced JSON, and transient network errors are
all absorbed here instead of crashing a run.

---

## 5. The tool system

### [app/tools/registry.py](app/tools/registry.py) — the toolbelt machinery

Defines three things:

- **`ToolContext`** — what every tool receives. Holds the `inputs` (list of `ExtractedInput`), the
  user `query`, and `upstream` (the previous step's output when chaining). Two helper methods
  decide what text a tool actually operates on:
  - `combined_text()` — joins all good inputs together (labeled by source), truncated to a budget.
  - `primary_text()` — **prefers `upstream`** (a chained step's output) if present, else the
    combined inputs, else the raw query. This single method is why chaining "just works": a tool
    automatically uses the previous step's output when there is one.
  - `truncate_for_llm(text)` — keeps the head (70%) and tail (25%) of over-long text with a
    `[truncated N chars]` marker in the middle, so prompts stay within budget without losing the
    start and end.
- **`Tool`** — a dataclass: `name`, `description`, `input_hint`, and `func` (the async function).
  The `description` is important — it's literally what the planner LLM reads to decide when to use
  the tool.
- **`ToolRegistry`** — a dict of tools with `register/get/names`, plus `describe_for_planner()`
  (formats all tools into the catalog text the planner sees) and `as_list()` (for the `/tools`
  endpoint).

At the bottom, `register_all_tools()` imports each tool module. **Importing a tool module is what
registers it** — each tool file ends with a `registry.register(Tool(...))` call that runs on
import. This is a clean plugin pattern: add a file, import it here, and the planner can use it.

### The six tools

Each tool is the same shape: a system prompt that **forces a strict output format**, an async
function that builds messages and calls `llm_client.chat(...)`, and a `registry.register(...)`
call. The strict formats are what make the outputs predictable (and testable).

| File | Tool | System prompt forces… | Notable detail |
|------|------|------------------------|----------------|
| [summarize.py](app/tools/summarize.py) | `summarize` | "One-line summary / Key points (3 bullets) / Detailed summary (5 sentences)" | Uses `primary_text()` so it summarizes either inputs *or* a chained transcript |
| [qa.py](app/tools/qa.py) | `qa` | Concise grounded answer, plain text | Only prepends "Context:" if the context differs from the question; the **general fallback** tool |
| [compare.py](app/tools/compare.py) | `compare` | Cross-source agree/disagree analysis, referencing `[source]` | Builds an explicitly **labeled** block per source so the model can cite them; needs ≥1 readable input |
| [code_explain.py](app/tools/code_explain.py) | `code_explain` | "What it does / Bugs / Time complexity (Big-O)" | Wraps the code in a fenced block; low temperature (0.1) for precision |
| [sentiment.py](app/tools/sentiment.py) | `sentiment` | "Sentiment / Confidence% / Justification" | Temperature 0.0 (deterministic); returns a neutral 0% result if no text |
| [youtube.py](app/tools/youtube.py) | `youtube_transcript` | (no LLM — fetches captions) | See below |

### [app/tools/youtube.py](app/tools/youtube.py) — the special one

This is the only tool that doesn't call the LLM; it fetches data from outside.

- **`find_youtube_urls(text)`** — a regex that recognizes every YouTube URL shape (`watch?v=`,
  `youtu.be/`, `shorts/`, `embed/`, `live/`, etc.) and extracts the 11-character video IDs,
  de-duplicated.
- **`_fetch_sync(video_id)`** — calls `youtube-transcript-api` and **carefully classifies
  failures** into distinct messages: captions disabled/missing, video unavailable/private, and —
  importantly — the "empty XML response" case (`"no element found"`), which usually means **the
  server's IP was blocked by YouTube** rather than the video lacking captions.
- **`fetch_transcript(video_id)`** — runs the blocking fetch in a thread
  (`asyncio.to_thread`) and retries once.
- **`youtube_transcript(ctx)`** — the tool entry. It searches the query **and all uploaded inputs**
  for URLs (so it can find a link *inside a PDF*), fetches each transcript, and returns them. The
  planner then typically chains `summarize` after this (`input_from: "step:1"`).

> The detailed "why did transcription fail and how to fix it" analysis lives in
> [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) §6.

---

## 6. The agent brain

This is the orchestration layer — the actual "agent." It decides *what* to do and *runs* it.

### [app/agent/planner.py](app/agent/planner.py) — deciding what to do

`make_plan(query, inputs, registry, clarification_history)` is where the LLM becomes an agent:

1. **`build_context_digest(inputs)`** — produces a compact summary of every input for the prompt:
   source, type, key metadata (OCR confidence, duration, OCR pages), and a truncated text preview.
   Inputs with errors are shown as `ERROR — ...`.
2. It fills the big **`_PLANNER_SYSTEM`** prompt with the tool catalog
   (`registry.describe_for_planner()`). That prompt tells the model the exact JSON shape to return
   and the rules: ask **one** clarifying question if the goal is ambiguous (don't guess), otherwise
   produce a **minimal ordered plan**, chain steps with `input_from: "step:N"`, prefer `context`
   when files are present, keep it to 1–6 steps. It even includes worked examples (including the
   YouTube-in-PDF case).
3. Calls `llm_client.chat_json(...)` at low temperature (0.1) to get the plan as JSON.
4. **Validates** the result: parses into a `Plan`, then **filters out any tool names the model
   hallucinated** (`s.tool in valid_names`). If the plan is empty but no clarification was
   requested, it falls back to a single `qa` step so the user always gets an answer.

### [app/agent/executor.py](app/agent/executor.py) — running the plan

`execute_plan(plan, query, inputs, registry, state, progress)` runs the steps in order:

- Caps the plan at `MAX_STEPS = 6`. For each step it creates a `StepResult`, sets it `running`,
  and (if a `progress` callback is supplied) reports live state.
- **Resolves the input**: if `input_from` is `"step:N"`, it looks up that earlier step's output
  from `outputs_by_step` and passes it as `ctx.upstream`. This is the mechanism behind chaining.
- Runs the tool via **`_run_with_retry`**, which wraps the call in a `StepTimer`, marks
  success/failure, **retries once** (with a 0.5s pause) on a generic exception, and — crucially —
  **does not retry `LLMNotConfiguredError`** (no point retrying a missing key). A failed step
  returns a readable `[Step failed: ...]` string rather than throwing, so the run continues.
- **`_pick_final_answer`** — chooses the answer to return: the **last successful step's** output
  (walking backwards), falling back to the last raw output, or a clear "couldn't complete" message
  if everything failed.

### [app/agent/orchestrator.py](app/agent/orchestrator.py) — tying it all together

The `Orchestrator.run(...)` method is the conductor. In order:

1. Creates/loads the `RunState` and stores the extracted inputs on it.
2. **Resumes a parked clarification** if one exists for this `conversation_id`: it pops the pending
   session and merges the original query with the user's new answer, so the follow-up reply
   continues the *original* task instead of starting over.
3. **Carries forward conversation memory**: `_carry_forward` pulls in earlier inputs from this
   conversation (relabeling them `(earlier)` and de-duplicating), and the last few transcript lines
   become `history`. This is what lets you ask "now summarize it" about a file you uploaded a turn
   ago.
4. Sets status to `planning`, calls `make_plan(...)`.
5. **If the plan needs clarification**: it parks a `PendingSession` in the session store, sets the
   clarifying question on the state, and returns *without executing* — the API surfaces the
   question to the user.
6. Otherwise it estimates cost, then calls `execute_plan(...)`.
7. **Persists memory**: appends the good new inputs (trimmed to a character budget by
   `_trim_to_budget`) and the user/assistant exchange into the conversation transcript (kept to the
   last 12 lines).

`_carry_forward` and `_trim_to_budget` are the two helpers that make multi-turn memory work without
the context growing forever.

### [app/agent/cost.py](app/agent/cost.py) — estimating spend

`estimate_cost(plan, query, inputs)` is a heuristic: it counts the characters of context, converts
to tokens at the rough `chars/4` rate, multiplies by the number of steps and fixed per-million
prices (input `$0.59`, output `$0.79`), and returns a `CostEstimate`. It's explicitly an estimate,
not billed usage.

### [app/agent/session_store.py](app/agent/session_store.py) — in-memory state

Three simple in-memory stores (plain dicts), plus their data classes:

- **`SessionStore`** — holds **parked clarifications** (`PendingSession`: the original query,
  inputs, the question asked, and history). `pop_pending` retrieves-and-removes in one call.
- **`RunStore`** — holds `RunState` objects by `run_id` so the `/runs/{id}/...` endpoints can look
  them up after the request.
- **`ConversationStore`** — holds per-conversation **memory** (`Conversation`: accumulated `inputs`
  and a rolling `transcript`). `reset()` exists mainly for tests.

> **Important limitation:** all three are in-process dicts. They don't survive a restart and don't
> work across multiple server instances. A production deployment would swap these for Redis or a DB.

---

## 7. The web layer

### [app/main.py](app/main.py) — the FastAPI app & endpoints

Sets up the app, CORS (open `*` for the demo), registers all tools at startup, and serves the
static UI. Endpoints:

- **`GET /`** and **`GET /static/{filename}`** — serve the chat UI (with a path-traversal guard:
  `Path(filename).name`).
- **`GET /health`** — reports whether the LLM and Whisper are configured, plus provider/model.
  Useful to confirm your keys are set.
- **`GET /tools`** — lists registered tools.
- **`POST /chat`** — *the* endpoint. Reads the form `message`, `conversation_id`, and uploaded
  `files`. `_ingest_uploads` reads and ingests all files **concurrently** with `asyncio.gather`.
  The typed message is added as a `user_query` input. It then calls `orchestrator.run(...)`. If an
  `LLMError` bubbles up (e.g. missing key), it's caught and returned as a clean error payload with
  HTTP 200 (so the UI can display it gracefully). `_serialize_state` shapes the response:
  `conversation_id`, `run_id`, `status`, `extracted_inputs`, `plan_trace`, `cost`, `final_answer`,
  `clarification`, `error`.
- **`GET /runs/{id}/status`** — returns the stored run state.
- **`GET /runs/{id}/stream`** — re-streams the already-computed `final_answer` token-by-token over
  **SSE** (Server-Sent Events), so the UI can show a typewriter effect.
- **`GET /runs/{id}/stream_live`** — streams tokens **directly from the LLM** as they're generated.

### [app/static/](app/static/) — the front end

`index.html`, `app.js`, and `style.css` make up a small single-page chat UI: a message box, a file
picker (multi-file), it POSTs to `/chat`, renders the extracted inputs + plan trace + answer, and
uses the SSE stream endpoint for the typing effect. (No framework — plain HTML/JS/CSS.)

---

## 8. The tests

### [tests/conftest.py](tests/conftest.py) — the safety-net fixture

One **autouse** fixture (`restore_registry`) runs around every test. Before each test it records
each tool's original `func`; after each test it **restores them**. This matters because the tests
monkey-patch tool functions (e.g. replace `fetch_transcript` with a fake) — without this fixture,
a patched function could leak into the next test. It also calls `conversation_store.reset()` so no
conversation memory bleeds between tests.

### [tests/test_sample_cases.py](tests/test_sample_cases.py) — 5 end-to-end scenarios

These are **integration tests**: they run the real FastAPI app via `TestClient` and exercise the
real ingestion path with **real generated file bytes**, but they **stub the external services**
(the LLM planner, the LLM chat, audio transcription, OCR, YouTube) with deterministic fakes via
`monkeypatch`. That isolates *our orchestration logic* from a live model.

Helper builders at the top:
- **`_wav_bytes()`** — synthesizes a real 1-second 440 Hz WAV in memory.
- **`_pdf_bytes(text)`** — builds a real PDF containing `text` using PyMuPDF.
- **`_png_bytes()`** — makes a blank PNG.
- **`_plan(*tools)`** — builds a fake planner JSON response with the given tools chained
  (`context` for step 1, `step:N` after), so a test can dictate exactly what plan the agent runs.
- **`_app_client()`** — returns a `TestClient` over the app.

The five tests (all passing — `pytest -q` → `..... [100%]`):

| Test | Flow it verifies | Key assertions |
|------|------------------|----------------|
| **TC1** `test_tc1_audio_transcription_and_summary` | Upload WAV → fake transcript → `summarize` | The 3-part summary format is present; ≥3 bullets; the plan's first tool is `summarize`; audio `duration_seconds` metadata survives into the response |
| **TC2** `test_tc2_pdf_natural_language_query` | Upload a **real PDF** → `qa` | PDF text is actually parsed (assert "alice" is in the extracted text); the answer mentions Alice/Bob/Carol; plan routes to `qa` |
| **TC3** `test_tc3_image_with_code` | Upload PNG → fake OCR returns code → `code_explain` | Output has "What it does / Bugs / Time complexity"; the OCR confidence (88.5) is reported in the input meta |
| **TC4** `test_tc4_pdf_youtube_url_chain` | PDF containing a YouTube URL → `youtube_transcript` → `summarize` | The two-step chain runs **in order** (`["youtube_transcript","summarize"]`); no clarification; 3-part summary present. *This proves tool chaining and URL-in-PDF detection.* |
| **TC5** `test_tc5_multi_file_unified_query` | Upload **WAV + PDF together** → `compare` | Plan routes to `compare`; the unified answer references the shared data ("12"); both `audio` and `pdf` appear in extracted-input types. *This proves multi-modal unified reasoning.* |

Each test asserts on the **public JSON contract** (`final_answer`, `plan_trace`,
`extracted_inputs`, `clarification`) — i.e. it tests the system the way a real client uses it.

---

## 9. How one request flows end-to-end

Putting it all together, here's the full path of `POST /chat` with a PDF that contains a YouTube
link and the message *"summarize the video in this PDF"*:

```
1. main.py /chat
   → reads the PDF bytes + message
   → _ingest_uploads()  ──▶  router.ingest_file()  ──▶  pdf_extractor.extract_pdf()
        → PyMuPDF reads the text (OCR fallback if scanned)  ──▶  ExtractedInput(type="pdf", text="...youtube.com/watch?v=...")
   → message becomes ExtractedInput(type="text", source="user_query")

2. orchestrator.run()
   → no parked clarification, no prior memory
   → planner.make_plan()
        → builds context digest + tool catalog
        → llm_client.chat_json()  ──▶  {"plan":[{youtube_transcript, context}, {summarize, step:1}]}
        → validates tool names

3. cost.estimate_cost()  ──▶  token/USD estimate

4. executor.execute_plan()
   → Step 1: youtube_transcript
        → ctx.combined_text() includes the PDF text
        → find_youtube_urls() finds the video id  ──▶  fetch_transcript()  ──▶  transcript text
        → outputs_by_step[1] = transcript
   → Step 2: summarize  (input_from="step:1")
        → ctx.upstream = transcript  ──▶  primary_text() uses it
        → llm_client.chat()  ──▶  3-part summary
   → _pick_final_answer()  ──▶  the summary (last successful step)

5. orchestrator persists conversation memory (inputs + transcript)

6. main.py _serialize_state()  ──▶  JSON: {final_answer, plan_trace, extracted_inputs, cost, ...}

7. UI calls /runs/{id}/stream  ──▶  SSE re-streams the answer token-by-token
```

Every layer degrades gracefully: a bad file becomes an `error` input, a failed step retries then
returns a readable message, a missing key produces a clean error payload — the request always
returns *something* useful, with a full trace of what happened.

---

*This walkthrough covers every `.py` file in `app/` and `tests/`. For the higher-level project
report (architecture diagram, feature list, test results, limitations, and the YouTube failure
deep-dive) see [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md).*
