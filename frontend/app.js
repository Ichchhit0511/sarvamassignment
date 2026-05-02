// =====================================================================
// Bike Troubleshooting Bot — frontend
// =====================================================================
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// Backend URL — set via <meta name="backend-url"> or leave blank to use
// same-origin /api/* (works for local dev and Netlify-with-proxy).
const BACKEND_URL = (
  document.querySelector('meta[name="backend-url"]')?.content || ""
).replace(/\/$/, "");
const api = (path) => BACKEND_URL + path;

// Stable session_id per browser, persisted in localStorage.
const SESSION_ID = (() => {
  let s = localStorage.getItem("bike_session_id");
  if (!s) {
    s = "web-" + crypto.randomUUID();
    localStorage.setItem("bike_session_id", s);
  }
  return s;
})();

// ---- Tab routing ------------------------------------------------------
$$(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    $$(".tab").forEach((b) => b.classList.remove("active"));
    $$(".tab-panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $(`#tab-${btn.dataset.tab}`).classList.add("active");
    if (btn.dataset.tab === "manuals") loadManuals();
    if (btn.dataset.tab === "health") loadHealth();
    if (btn.dataset.tab === "dashboard") loadDashboard();
  });
});

// ---- Helpers ----------------------------------------------------------
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

async function parseResponseOrThrow(response) {
  const contentType = response.headers.get("content-type") || "";
  const bodyText = await response.text();

  let data = null;
  if (contentType.includes("application/json")) {
    try {
      data = JSON.parse(bodyText);
    } catch (err) {
      throw new Error(bodyText || "Invalid JSON response from server.");
    }
  }

  if (!response.ok) {
    const detail = data?.detail || data?.error || bodyText || `Request failed with status ${response.status}`;
    throw new Error(String(detail));
  }

  if (data != null) return data;

  try {
    return JSON.parse(bodyText);
  } catch (err) {
    throw new Error(bodyText || "Invalid response from server.");
  }
}

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html != null) e.innerHTML = html;
  return e;
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function fmtTime(ts) {
  const d = new Date(ts);
  return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// =======================================================================
// CHAT TAB
// =======================================================================
const chatLog = $("#chat-log");
const queryInput = $("#query-input");
const imageInput = $("#image-input");
const imagePill = $("#image-pill");
let pendingImage = null;

imageInput.addEventListener("change", async () => {
  const f = imageInput.files[0];
  if (!f) { pendingImage = null; imagePill.classList.add("hidden"); return; }
  pendingImage = await fileToBase64(f);
  imagePill.textContent = `📷 ${f.name}`;
  imagePill.classList.remove("hidden");
});

function addMessage(role, html) {
  const wrap = el("div", `msg ${role}`);
  wrap.appendChild(el("div", "bubble", html));
  chatLog.appendChild(wrap);
  chatLog.scrollTop = chatLog.scrollHeight;
  return wrap;
}

function renderAnswer(resp) {
  const a = resp.answer;
  let html = `<div>${escapeHtml(a.answer)}</div>`;
  if (a.manual_supported && a.citations && a.citations.length) {
    const pages = a.citations.map((c) => c.page).filter(Boolean);
    const uniq = [...new Set(pages)];
    if (uniq.length === 1) {
      html += `<div class="citations">📖 You can refer to page ${uniq[0]} of the manual for more details.</div>`;
    } else if (uniq.length > 1) {
      html += `<div class="citations">📖 You can refer to pages ${uniq.join(", ")} of the manual for more details.</div>`;
    }
  } else if (!a.manual_supported) {
    html += `<div class="citations">This is not covered in the manual. Please contact an authorized service center.</div>`;
  }
  return html;
}

$("#chat-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = queryInput.value.trim();
  if (!text && !pendingImage) return;

  const manualId = $("#manual-select").value;
  if (!manualId) {
    addMessage("bot", "Please ingest a manual first (Manuals tab).");
    return;
  }

  let userHtml = "";
  if (pendingImage) userHtml += `<img src="${pendingImage}" />`;
  if (text) userHtml += escapeHtml(text);
  addMessage("user", userHtml);

  queryInput.value = "";
  const sentImage = pendingImage;
  pendingImage = null;
  imageInput.value = "";
  imagePill.classList.add("hidden");

  const thinking = addMessage("bot", "🤔 thinking…");

  try {
    const r = await fetch(api("/api/query"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        manual_id: manualId,
        query: text || "Help me troubleshoot the issue in this photo.",
        image_b64: sentImage,
        session_id: SESSION_ID,
      }),
    });
    const data = await parseResponseOrThrow(r);
    thinking.querySelector(".bubble").innerHTML = renderAnswer(data);
  } catch (err) {
    thinking.querySelector(".bubble").textContent = "Error: " + err.message;
  }
});

$("#clear-memory").addEventListener("click", async () => {
  await fetch(api("/api/memory/clear"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: SESSION_ID }),
  });
  chatLog.innerHTML = "";
  addMessage("bot", "🧠 Memory cleared. New conversation starts now.");
});

// =======================================================================
// MANUALS TAB
// =======================================================================
async function loadManuals() {
  const r = await fetch(api("/api/manuals"));
  const data = await parseResponseOrThrow(r);
  const tbody = $("#manuals-table tbody");
  tbody.innerHTML = "";
  const sel = $("#manual-select");
  sel.innerHTML = "";
  if (!data.manuals.length) {
    tbody.appendChild(el("tr", null, '<td colspan="2" class="muted">No manuals yet.</td>'));
    sel.appendChild(el("option", null, "(none — ingest a PDF first)"));
    return;
  }
  data.manuals.forEach((m) => {
    tbody.appendChild(el("tr", null,
      `<td>${escapeHtml(m.manual_id)}</td><td>${m.chunk_count}</td>`));
    const opt = document.createElement("option");
    opt.value = m.manual_id;
    opt.textContent = m.manual_id;
    sel.appendChild(opt);
  });
}

$("#refresh-manuals").addEventListener("click", loadManuals);

$("#ingest-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData();
  fd.append("manual_id", $("#manual-id").value.trim());
  fd.append("file", $("#manual-file").files[0]);
  $("#ingest-status").textContent = "Ingesting… this can take a minute for large PDFs.";
  try {
    const r = await fetch(api("/api/ingest"), { method: "POST", body: fd });
    const data = await parseResponseOrThrow(r);
    $("#ingest-status").textContent = data.ok
      ? `✅ Ingested ${data.chunks} chunks across ${data.pages} pages.`
      : `❌ ${JSON.stringify(data)}`;
    loadManuals();
  } catch (err) {
    $("#ingest-status").textContent = "❌ " + err.message;
  }
});

// =======================================================================
// DASHBOARD TAB
// =======================================================================
const charts = {};

function makeOrUpdateChart(id, type, data, options = {}) {
  const ctx = document.getElementById(id);
  if (charts[id]) {
    charts[id].data = data;
    charts[id].options = options;
    charts[id].update();
    return;
  }
  charts[id] = new Chart(ctx, { type, data, options });
}

async function loadDashboard() {
  const win = $("#dash-window").value;
  const [aggR, recentR] = await Promise.all([
    fetch(api(`/api/metrics?window_hours=${win}`)).then((r) => r.json()),
    fetch(api(`/api/metrics/recent?limit=30`)).then((r) => r.json()),
  ]);

  // KPIs
  $("#kpi-total").textContent = aggR.total_queries;
  $("#kpi-supported").textContent = (aggR.manual_supported_rate * 100).toFixed(1) + "%";
  $("#kpi-cites").textContent = (aggR.citations_kept_rate * 100).toFixed(1) + "%";
  $("#kpi-score").textContent = aggR.avg_top_score.toFixed(3);
  $("#kpi-latency").textContent = (aggR.avg_total_ms / 1000).toFixed(2) + " s";

  // Stage latency bar chart
  makeOrUpdateChart("chart-latency", "bar", {
    labels: ["Vision", "Rewrite", "Retrieve", "Generate", "Verify"],
    datasets: [{
      label: "ms",
      data: [
        aggR.stage_latency_ms.vision || 0,
        aggR.stage_latency_ms.rewrite || 0,
        aggR.stage_latency_ms.retrieve || 0,
        aggR.stage_latency_ms.generate || 0,
        aggR.stage_latency_ms.verify || 0,
      ],
      backgroundColor: ["#a78bfa", "#7c3aed", "#5b3df5", "#4338ca", "#312e81"],
    }],
  }, { responsive: true, plugins: { legend: { display: false } } });

  // Token usage stacked bar
  makeOrUpdateChart("chart-tokens", "bar", {
    labels: ["Sarvam", "Gemini"],
    datasets: [
      {
        label: "Input tokens",
        data: [aggR.avg_sarvam_input_tokens, aggR.avg_gemini_input_tokens],
        backgroundColor: "#5b3df5",
      },
      {
        label: "Output tokens",
        data: [aggR.avg_sarvam_output_tokens, aggR.avg_gemini_output_tokens],
        backgroundColor: "#a78bfa",
      },
    ],
  }, { responsive: true, scales: { x: { stacked: true }, y: { stacked: true } } });

  const totalIn = aggR.avg_sarvam_input_tokens + aggR.avg_gemini_input_tokens;
  const totalOut = aggR.avg_sarvam_output_tokens + aggR.avg_gemini_output_tokens;
  $("#token-cost-hint").textContent =
    `Avg ${totalIn} input + ${totalOut} output tokens per query.`;

  // Confidence donut
  makeOrUpdateChart("chart-confidence", "doughnut", {
    labels: ["high", "medium", "low"],
    datasets: [{
      data: [aggR.confidence.high, aggR.confidence.medium, aggR.confidence.low],
      backgroundColor: ["#16a34a", "#d97706", "#dc2626"],
    }],
  }, { responsive: true, plugins: { legend: { position: "bottom" } } });

  // Languages bar
  const langLabels = Object.keys(aggR.languages);
  const langValues = Object.values(aggR.languages);
  makeOrUpdateChart("chart-languages", "bar", {
    labels: langLabels.length ? langLabels : ["(no data)"],
    datasets: [{
      label: "queries",
      data: langValues.length ? langValues : [0],
      backgroundColor: "#5b3df5",
    }],
  }, { responsive: true, plugins: { legend: { display: false } }, indexAxis: "y" });

  // Recent table
  const tbody = $("#recent-table tbody");
  tbody.innerHTML = "";
  for (const q of recentR.queries) {
    const conf = (q.confidence || "low").toLowerCase();
    const confCls = conf === "high" ? "conf-high" : conf === "medium" ? "conf-medium" : "conf-low";
    tbody.appendChild(el("tr", null, `
      <td>${fmtTime(q.ts)}</td>
      <td class="q-cell" title="${escapeHtml(q.query || "")}">${escapeHtml(q.query || "")}</td>
      <td>${escapeHtml(q.language || "—")}</td>
      <td>${q.has_image ? "📷" : ""}</td>
      <td><span class="conf-pill ${confCls}">${conf}</span></td>
      <td>${(q.top_retrieval_score || 0).toFixed(3)}</td>
      <td>${q.num_citations_kept || 0}/${q.num_citations_raw || 0}</td>
      <td>${q.sarvam_input_tokens || 0}/${q.sarvam_output_tokens || 0}</td>
      <td>${q.gemini_input_tokens || 0}/${q.gemini_output_tokens || 0}</td>
      <td>${q.total_ms || 0}</td>
    `));
  }
  if (!recentR.queries.length) {
    tbody.appendChild(el("tr", null, '<td colspan="10" class="muted">No queries yet — ask one in the Chat tab.</td>'));
  }
}

$("#dash-refresh").addEventListener("click", loadDashboard);
$("#dash-window").addEventListener("change", loadDashboard);

// =======================================================================
// HEALTH TAB
// =======================================================================
async function loadHealth() {
  const r = await fetch(api("/api/health"));
  const d = await r.json();
  const ul = $("#health-list");
  ul.innerHTML = "";
  const items = [
    ["Sarvam (final answer LLM)", d.sarvam_key_set],
    ["Gemini (vision + embeddings + rewriter)", d.gemini_key_set],
    ["WhatsApp", d.whapi_token_set],
  ];
  items.forEach(([label, ok]) => {
    const li = el("li", null,
      `<span>${label}</span><span class="${ok ? "dot-ok" : "dot-no"}">${ok ? "● ready" : "○ not set"}</span>`);
    ul.appendChild(li);
  });
}

// ---- Initial load ----------------------------------------------------
loadManuals();
