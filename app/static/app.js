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
  const reviewToggle = document.getElementById("reviewToggle");

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

  // ---- Stage 1: submit -> /plan --------------------------------------------

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

    let plan;
    try {
      const resp = await fetch("/plan", { method: "POST", body: form });
      if (!resp.ok) {
        const body = await resp.text();
        throw new Error(`${resp.status} ${resp.statusText}: ${body.slice(0, 120)}`);
      }
      plan = await resp.json();
    } catch (e) {
      bubble.innerHTML = `<div class="answer err">planning failed: ${esc(String(e))}</div>`;
      sendBtn.disabled = false; return;
    }

    conversationId = plan.conversation_id || conversationId;
    localStorage.setItem("conversation_id", conversationId);

    const header = renderManifest(plan.input_manifest, plan.detected_inputs)
      + renderInputsSection(plan.extracted_inputs);

    if (plan.error) {
      bubble.innerHTML = header + `<div class="answer err">${esc(plan.error)}</div>`;
      sendBtn.disabled = false; return;
    }

    if (plan.needs_clarification) {
      shell.classList.add("clarifying");
      bubble.innerHTML = header + `<div class="clarify">${esc(plan.clarifying_question)}</div>`;
      scrollDown(); sendBtn.disabled = false; input.focus();
      return;
    }

    if (reviewToggle && reviewToggle.checked) {
      renderPlanEditor(plan, shell, header, text);
      sendBtn.disabled = false;
    } else {
      runPlan(plan.plan_id, plan.steps, shell, header, text);
    }
  }

  // ---- Editable plan review ------------------------------------------------

  function renderPlanEditor(plan, shell, headerHtml, originalText) {
    const bubble = shell.querySelector(".bubble");
    const steps = (plan.steps || []).map((s) => ({ ...s }));
    const tools = plan.available_tools || [];
    let nextId = steps.reduce((m, s) => Math.max(m, s.id || 0), 0) + 1;

    const wrap = document.createElement("div");
    bubble.innerHTML = headerHtml;
    bubble.appendChild(wrap);

    function toolOptions(sel) {
      return tools.map((t) =>
        `<option value="${esc(t.name)}"${t.name === sel ? " selected" : ""}>${esc(t.name)}</option>`).join("");
    }
    function depLabel(step) {
      if (!step.input_from || !step.input_from.startsWith("step:")) {
        return step.input_from === "query" ? "uses your message" : "uses all inputs";
      }
      const refId = parseInt(step.input_from.split(":")[1], 10);
      const pos = steps.findIndex((s) => s.id === refId);
      return pos >= 0 ? `uses step ${pos + 1}` : "uses all inputs";
    }

    function rerender() {
      wrap.innerHTML = `<div class="plan-editor">
        <div class="plan-editor-head">The planner proposed these steps. Edit instructions, reorder, add or remove steps, then run.</div>
        <ul class="plan-steps"></ul>
        <div class="add-step">
          <select class="add-tool">${toolOptions(tools[0] && tools[0].name)}</select>
          <input class="add-instr" type="text" placeholder="what this step should do (optional)" />
          <button class="add-btn" type="button">+ add step</button>
        </div>
        <div class="plan-actions">
          <button class="run-plan send">Run this plan</button>
          <button class="cancel-plan ghost">Cancel</button>
        </div>
      </div>`;
      const ul = wrap.querySelector(".plan-steps");

      steps.forEach((s, idx) => {
        const li = document.createElement("li");
        li.className = "plan-step-card";
        li.draggable = true;
        li.innerHTML = `
          <div class="psc-head">
            <span class="drag" title="Drag to reorder">&#x28FF;</span>
            <span class="psc-num">${idx + 1}</span>
            <select class="psc-tool">${toolOptions(s.tool)}</select>
            <span class="psc-dep">${depLabel(s)}</span>
            <button class="psc-del" title="Remove step">&#x2715;</button>
          </div>
          <div class="psc-desc">${esc(s.description || "")}</div>
          <textarea class="psc-instr" rows="1" placeholder="add an instruction for this step (optional)">${esc(s.instructions || "")}</textarea>`;

        li.querySelector(".psc-tool").addEventListener("change", (e) => { s.tool = e.target.value; });
        li.querySelector(".psc-instr").addEventListener("input", (e) => { s.instructions = e.target.value; });
        li.querySelector(".psc-del").addEventListener("click", () => { steps.splice(idx, 1); rerender(); });

        li.addEventListener("dragstart", (e) => {
          li.classList.add("dragging");
          e.dataTransfer.effectAllowed = "move";
          e.dataTransfer.setData("text/plain", String(idx));
        });
        li.addEventListener("dragend", () => li.classList.remove("dragging"));
        li.addEventListener("dragover", (e) => e.preventDefault());
        li.addEventListener("drop", (e) => {
          e.preventDefault();
          const from = parseInt(e.dataTransfer.getData("text/plain"), 10);
          if (Number.isNaN(from) || from === idx) return;
          const [moved] = steps.splice(from, 1);
          steps.splice(idx, 0, moved);
          rerender();
        });
        ul.appendChild(li);
      });

      wrap.querySelector(".add-btn").addEventListener("click", () => {
        const tool = wrap.querySelector(".add-tool").value;
        const instr = wrap.querySelector(".add-instr").value.trim();
        steps.push({ id: nextId++, tool, description: "(step you added)", input_from: "context", instructions: instr });
        rerender();
      });
      wrap.querySelector(".run-plan").addEventListener("click", () => {
        runPlan(plan.plan_id, steps, shell, headerHtml, originalText);
      });
      wrap.querySelector(".cancel-plan").addEventListener("click", () => {
        wrap.innerHTML = `<div class="answer muted">Cancelled — nothing was run.</div>`;
      });
    }
    rerender();
    scrollDown();
  }

  // ---- Stage 2: /execute -> live SSE stream --------------------------------

  async function runPlan(planId, steps, shell, headerHtml, originalText) {
    const bubble = shell.querySelector(".bubble");
    sendBtn.disabled = true;

    let exec;
    try {
      const resp = await fetch("/execute", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ plan_id: planId, conversation_id: conversationId, steps }),
      });
      if (resp.status === 404) {
        bubble.innerHTML = headerHtml + `<div class="answer err">This plan expired. Send your message again to re-plan.</div>`;
        sendBtn.disabled = false; return;
      }
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      exec = await resp.json();
    } catch (e) {
      bubble.innerHTML = headerHtml + `<div class="answer err">could not start execution: ${esc(String(e))}</div>`;
      sendBtn.disabled = false; return;
    }

    const runId = exec.run_id;
    const warnings = exec.warnings || [];
    const warnHtml = warnings.length
      ? `<div class="warnings">${warnings.map((w) => `<div>&#9888; ${esc(w)}</div>`).join("")}</div>`
      : "";

    bubble.innerHTML = headerHtml + warnHtml
      + `<details class="section" open><summary>plan &middot; running</summary>
           <div class="body"><ul class="steps" id="steps-${runId}"></ul></div></details>`
      + `<div class="answer" id="answer-${runId}"></div>`;

    const stepsEl = document.getElementById(`steps-${runId}`);
    const answerEl = document.getElementById(`answer-${runId}`);
    streamExecution(runId, planId, stepsEl, answerEl, shell, headerHtml, originalText);
  }

  function renderLiveSteps(el, steps) {
    el.innerHTML = (steps || []).map((s) => `
      <li class="step">
        ${statusToken(s.status)}
        <div class="detail">
          <div class="tool">${s.id}. ${esc(s.tool)}<span class="dur">${s.duration_ms || 0}ms</span></div>
          <div class="reason">${esc(s.description || "")}</div>
          ${s.output_preview ? `<div class="out">${esc(s.output_preview)}${s.output_preview.length >= 240 ? "..." : ""}</div>` : ""}
          ${s.error ? `<div class="out err">${esc(s.error)}</div>` : ""}
        </div>
      </li>`).join("");
  }

  function streamExecution(runId, planId, stepsEl, answerEl, shell, headerHtml, originalText) {
    const es = new EventSource(`/runs/${runId}/execute_stream?plan_id=${encodeURIComponent(planId)}`);
    let failedSteps = [];

    es.onmessage = (e) => {
      let d;
      try { d = JSON.parse(e.data); } catch (_) { return; }

      if (d.type === "progress") {
        renderLiveSteps(stepsEl, d.steps);
        failedSteps = (d.steps || []).filter((s) => s.status === "failure");
      } else if (d.type === "answer") {
        answerEl.textContent = d.final_answer || "(no answer)";
        if (d.any_failed && failedSteps.length) {
          offerReplan(answerEl, shell, headerHtml, originalText, failedSteps);
        }
      } else if (d.type === "error") {
        answerEl.className = "answer err";
        answerEl.textContent = d.message;
      } else if (d.type === "done") {
        es.close();
        sendBtn.disabled = false;
      }
      scrollDown();
    };
    es.onerror = () => { es.close(); sendBtn.disabled = false; };
  }

  // ---- Failure -> re-plan --------------------------------------------------

  function offerReplan(answerEl, shell, headerHtml, originalText, failedSteps) {
    const note = failedSteps
      .map((s) => `Step '${s.tool}' failed: ${s.error || "unknown error"}.`)
      .join(" ") + " Adjust the plan to work around this.";

    const btn = document.createElement("button");
    btn.className = "replan-btn ghost";
    btn.textContent = "Re-plan around this";
    btn.addEventListener("click", () => replan(shell, headerHtml, originalText, note, btn));
    answerEl.insertAdjacentElement("afterend", btn);
  }

  async function replan(shell, headerHtml, originalText, failureNote, btn) {
    btn.disabled = true;
    const bubble = shell.querySelector(".bubble");
    const notice = document.createElement("div");
    notice.className = "pending-line";
    notice.textContent = "re-planning…";
    bubble.appendChild(notice);

    const form = new FormData();
    form.append("message", originalText || "");
    form.append("conversation_id", conversationId);
    form.append("replan_notes", failureNote);

    let plan;
    try {
      const resp = await fetch("/plan", { method: "POST", body: form });
      if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
      plan = await resp.json();
    } catch (e) {
      notice.className = "answer err";
      notice.textContent = `re-plan failed: ${esc(String(e))}`;
      return;
    }

    conversationId = plan.conversation_id || conversationId;
    const header = renderManifest(plan.input_manifest, plan.detected_inputs)
      + renderInputsSection(plan.extracted_inputs);

    if (plan.needs_clarification) {
      bubble.innerHTML = header + `<div class="clarify">${esc(plan.clarifying_question)}</div>`;
      return;
    }
    // Always show the editor for a re-plan so the user can review the new route.
    renderPlanEditor(plan, shell, header, originalText);
  }

  sendBtn.addEventListener("click", send);
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  });
})();
