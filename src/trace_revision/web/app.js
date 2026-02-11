// app.js (shared across pages)

function qs(id){ return document.getElementById(id); }

function showError(msg){
  const box = qs("errorBox");
  if(!box) return;
  box.style.display = "block";
  box.textContent = msg;
}

function hideError(){
  const box = qs("errorBox");
  if(!box) return;
  box.style.display = "none";
  box.textContent = "";
}

function saveDraftToSession(sourceText, targetText){
  sessionStorage.setItem("trace_source_text", sourceText ?? "");
  sessionStorage.setItem("trace_target_text", targetText ?? "");
}

function loadDraftFromSession(){
  return {
    source: sessionStorage.getItem("trace_source_text") ?? "",
    target: sessionStorage.getItem("trace_target_text") ?? ""
  };
}

function saveResultsToSession(rows){
  sessionStorage.setItem("trace_rows", JSON.stringify(rows ?? []));
}

function loadResultsFromSession(){
  const raw = sessionStorage.getItem("trace_rows");
  if(!raw) return [];
  try { return JSON.parse(raw); } catch { return []; }
}

async function postJSON(url, data){
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data)
  });
  if(!res.ok){
    const txt = await res.text().catch(()=> "");
    throw new Error(`HTTP ${res.status} ${res.statusText}\n${txt}`);
  }
  return await res.json();
}

// -------- index page wiring --------
function initIndex(){
  const sourceEl = qs("sourceText");
  const targetEl = qs("targetText");
  const btn = qs("analyzeBtn");

  if(!sourceEl || !targetEl || !btn) return;

  // restore prior text
  const {source, target} = loadDraftFromSession();
  if(source) sourceEl.value = source;
  if(target) targetEl.value = target;

  // IMPORTANT: ensure click handler attaches and fires
  btn.addEventListener("click", async (e) => {
    e.preventDefault();
    hideError();

    const src = sourceEl.value ?? "";
    const tgt = targetEl.value ?? "";

    if(!src.trim() || !tgt.trim()){
      showError("Please provide both Source Text and Target Text.");
      return;
    }

    saveDraftToSession(src, tgt);

    // navigate immediately to processing page (so user sees movement)
    window.location.href = "/processing";
  });
}

// -------- processing page wiring --------
function initProcessing(){
  const tickerEl = qs("ticker");
  if(!tickerEl) return;

  // animate ticker: 1..5 arrows then loop
  let k = 1;
  setInterval(() => {
    k = (k % 5) + 1;
    tickerEl.textContent = ">".repeat(k);
  }, 350);

  // run analysis (using stored texts)
  (async () => {
    try{
      const {source, target} = loadDraftFromSession();
      const data = await postJSON("/api/analyze", {
        source_text: source,
        target_text: target
      });
      saveResultsToSession(data.rows || []);
      window.location.href = "/results";
    }catch(err){
      // If processing fails, show a readable error and provide a back link
      const msg = (err && err.message) ? err.message : String(err);
      document.body.innerHTML = `
        <div class="processing-wrap">
          <div class="processing-card">
            <h1 style="color:#fb7185">Analysis failed</h1>
            <pre class="error" style="display:block">${msg.replaceAll("<","&lt;")}</pre>
            <button class="btn ghost" onclick="window.location.href='/'">← Back</button>
          </div>
        </div>
      `;
    }
  })();
}

// -------- results page wiring --------
function renderTable(rows){
  const table = qs("resultsTable");
  if(!table) return;

  if(!rows || rows.length === 0){
    table.innerHTML = `<tr><td style="padding:14px;color:rgba(231,246,255,0.7)">No rows returned.</td></tr>`;
    return;
  }

  const cols = Object.keys(rows[0]);
  const thead = `<thead><tr>${cols.map(c => `<th>${c}</th>`).join("")}</tr></thead>`;
  const tbody = `<tbody>${
    rows.map(r => `<tr>${cols.map(c => {
      const v = r[c];
      const s = (v === null || v === undefined) ? "" : String(v);
      // show spans neatly
      const cell = (Array.isArray(v)) ? `<code>[${v[0]}, ${v[1]}]</code>` : s;
      return `<td>${cell}</td>`;
    }).join("")}</tr>`).join("")
  }</tbody>`;
  table.innerHTML = thead + tbody;
}

function initResults(){
  const backBtn = qs("backBtn");
  const exportBtn = qs("exportBtn");
  if(!backBtn || !exportBtn) return;

  const rows = loadResultsFromSession();
  renderTable(rows);

  backBtn.addEventListener("click", () => {
    window.location.href = "/";
  });

  exportBtn.addEventListener("click", async () => {
    try{
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ rows })
      });
      if(!res.ok){
        const t = await res.text().catch(()=> "");
        throw new Error(`Export failed: HTTP ${res.status}\n${t}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "trace_results.csv";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }catch(err){
      showError(err.message || String(err));
    }
  });
}

// run correct initializer depending on page
document.addEventListener("DOMContentLoaded", () => {
  initIndex();
  initProcessing();
  initResults();
});
