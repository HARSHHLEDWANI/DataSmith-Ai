(() => {
  "use strict";

  const chat = document.getElementById("chat");
  const welcome = document.getElementById("welcome");
  const input = document.getElementById("messageInput");
  const sendBtn = document.getElementById("sendBtn");
  const fileInput = document.getElementById("fileInput");
  const fileChips = document.getElementById("fileChips");
  const dropzone = document.getElementById("dropzone");
  const healthEl = document.getElementById("health");
  const composerError = document.getElementById("composerError");

  let pendingFiles = [];
  let conversationId = localStorage.getItem("conversation_id") || "";

  function showError(msg) {
    composerError.textContent = msg;
    composerError.hidden = false;
  }
  function clearError() { composerError.hidden = true; composerError.textContent = ""; }
  function esc(s) {
    return (s || "").replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  function statusToken(s) {
    const tokens = {
      success: "&#x2713; done",
      running: '<span class="pulse">&middot;</span> running',
      failure: "&#x2717; failed",
      skipped: "&ndash; skipped",
      pending: "&middot; pending",
    };
    return `<span class="st st-${s}">${tokens[s] || tokens.pending}</span>`;
  }

  fetch("/health")
    .then((r) => {
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      return r.json();
    })
    .then((h) => {
      if (h.llm_configured) {
        healthEl.textContent = `${h.provider} / ${h.model}`;
        healthEl.classList.add("ok");
      } else {
        healthEl.textContent = "LLM_API_KEY not set";
        healthEl.classList.add("warn");
      }
    })
    .catch((err) => {
      healthEl.textContent = `offline (${err.message})`;
      healthEl.classList.add("warn");
    });

  function renderChips() {
    fileChips.innerHTML = "";
    pendingFiles.forEach((f, i) => {
      const chip = document.createElement("div");
      chip.className = "chip";
      const kb = (f.size / 1024).toFixed(0);
      chip.innerHTML = `<span>${esc(f.name)} ${kb}kb</span><button title="Remove">remove</button>`;
      chip.querySelector("button").onclick = () => { pendingFiles.splice(i, 1); renderChips(); };
      fileChips.appendChild(chip);
    });
  }
  function addFiles(list) {
    clearError();
    for (const f of list) {
      if (f.size > 25 * 1024 * 1024) { showError(`${f.name} is over the 25mb limit`); continue; }
      pendingFiles.push(f);
    }
    renderChips();
  }
  fileInput.addEventListener("change", (e) => { addFiles(e.target.files); fileInput.value = ""; });

  ["dragover", "dragenter"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));
  dropzone.addEventListener("drop", (e) => { if (e.dataTransfer?.files) addFiles(e.dataTransfer.files); });

  input.addEventListener("input", () => {
    input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px";
  });
  document.querySelectorAll(".chip-example").forEach((b) =>
    b.addEventListener("click", () => { input.value = b.dataset.fill; input.focus(); }));

  function addUserMsg(text, files) {
    if (welcome) welcome.style.display = "none";
    const fileLine = files.length ? `\nattached: ${files.map((f) => f.name).join(", ")}` : "";
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble">${esc(text + fileLine)}</div>`;
    chat.appendChild(el); scrollDown();
  }

  function agentShell() {
    const el = document.createElement("div");
    el.className = "msg agent";
    el.innerHTML = `<div class="bubble"></div>`;
    chat.appendChild(el); scrollDown();
    return el;
  }

  function renderManifest(manifest, refs) {
    if (!manifest) return "";
    const refLines = (refs || []).map((r) => {
      const label = r.kind === "youtube" ? `youtube link ${r.value}` : r.url;
      const via = r.via === "pdf_annotation" ? ", clickable link" : "";
      return `<span class="ref">&#8627; ${esc(label)} (inside ${esc(r.found_in)}${via})</span>`;
    }).join("");
    return `<div class="manifest">${esc(manifest)}${refLines}</div>`;
  }

  function renderInputsSection(inputs) {
    if (!inputs || !inputs.length) return "";
    const cards = inputs.map((i) => {
      const badges = [];
      if (i.meta && i.meta.ocr_confidence !== undefined)
        badges.push(`<span class="badge">ocr ${i.meta.ocr_confidence}%</span>`);
      if (i.meta && i.meta.duration_seconds !== undefined)
        badges.push(`<span class="badge">${i.meta.duration_seconds}s</span>`);
      if (i.meta && i.meta.ocr_pages && i.meta.ocr_pages.length)
        badges.push(`<span class="badge">ocr pages ${i.meta.ocr_pages.join(",")}</span>`);
      if (i.meta && i.meta.pages) badges.push(`<span class="badge">${i.meta.pages}p</span>`);
      if (i.error) badges.push(`<span class="badge err">error</span>`);
      const content = i.error ? esc(i.error) : esc(i.text || "(no text)");
      return `<div class="input-card"><h4>${esc(i.source)} <span class="badge">${i.type}</span></h4>
        <div class="meta">${badges.join(" ")}</div><pre>${content}</pre></div>`;
    }).join("");
    return `<details class="section"><summary>extracted input (${inputs.length})</summary>
      <div class="body">${cards}</div></details>`;
  }

  function renderTrace(steps, cost) {
    const items = (steps || []).map((s) => `
      <li class="step">
        ${statusToken(s.status)}
        <div class="detail">
          <div class="tool">${s.step}. ${esc(s.tool)}<span class="dur">${s.duration_ms || 0}ms</span></div>
          <div class="reason">${esc(s.reasoning || "")}</div>
          ${s.output ? `<div class="out">${esc((s.output || "").slice(0, 240))}${s.output.length > 240 ? "..." : ""}</div>` : ""}
          ${s.error ? `<div class="out err">${esc(s.error)}</div>` : ""}
        </div>
      </li>`).join("");
    const costLine = cost
      ? `<div class="cost">~${cost.estimated_input_tokens + cost.estimated_output_tokens} tokens · ~$${cost.estimated_usd.toFixed(5)} · ${esc(cost.note)}</div>`
      : "";
    return `<details class="section" open><summary>plan (${(steps || []).length} steps)</summary>
      <div class="body"><ul class="steps">${items || '<li class="pending-line">planning</li>'}</ul>${costLine}</div></details>`;
  }

  async function send() {
    const text = input.value.trim();
    if (!text && pendingFiles.length === 0) return;
    clearError();
    const files = pendingFiles.slice();
    addUserMsg(text, files);
    input.value = ""; input.style.height = "auto";
    pendingFiles = []; renderChips();
    sendBtn.disabled = true;

    const shell = agentShell();
    const bubble = shell.querySelector(".bubble");
    bubble.innerHTML = `<div class="pending-line">planning</div>`;

    const form = new FormData();
    form.append("message", text);
    form.append("conversation_id", conversationId);
    files.forEach((f) => form.append("files", f));

    let data;
    try {
      const resp = await fetch("/chat", { method: "POST", body: form });
      if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`${resp.status} ${resp.statusText}: ${body.slice(0, 120)}`);
      }
      data = await resp.json();
    } catch (e) {
      bubble.innerHTML = `<div class="answer err">request failed: ${esc(String(e))}</div>`;
      sendBtn.disabled = false; return;
    }

    conversationId = data.conversation_id || conversationId;
    localStorage.setItem("conversation_id", conversationId);

    let html = renderManifest(data.input_manifest, data.detected_inputs);
    html += renderInputsSection(data.extracted_inputs);

    if (data.clarification) {
      shell.classList.add("clarifying");
      html += renderTrace(data.plan_trace, data.cost);
      html += `<div class="clarify">${esc(data.clarification)}</div>`;
      bubble.innerHTML = html;
      scrollDown(); sendBtn.disabled = false; input.focus();
      return;
    }

    html += renderTrace(data.plan_trace, data.cost);
    html += `<div class="answer" id="answer-${data.run_id}"></div>`;
    bubble.innerHTML = html;
    const answerEl = document.getElementById(`answer-${data.run_id}`);

    if (data.error && !data.final_answer) {
      answerEl.className = "answer err";
      answerEl.textContent = data.error;
      sendBtn.disabled = false; return;
    }

    if (data.run_id) {
      streamAnswer(data.run_id, answerEl, data.final_answer);
    } else {
      answerEl.textContent = data.final_answer || "(no answer)";
    }
    scrollDown();
    sendBtn.disabled = false;
  }

  function streamAnswer(runId, el, fallback) {
    let acc = "";
    const es = new EventSource(`/runs/${runId}/stream`);
    es.onmessage = (e) => {
      try {
        const d = JSON.parse(e.data);
        if (d.token) { acc += d.token; el.textContent = acc; scrollDown(); }
        if (d.done) { es.close(); if (!acc) el.textContent = fallback || ""; }
        if (d.error) { es.close(); el.textContent = fallback || d.error; }
      } catch (_) {}
    };
    es.onerror = () => { es.close(); if (!acc) el.textContent = fallback || ""; };
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
})();
