// Dashboard client: plain JS + inline SVG, no external libraries, no network beyond this server.
"use strict";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const state = { meta: null, frontier: null, rec: null, es: null };

async function api(path) {
  const r = await fetch(path);
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || r.status);
  return j;
}

// A number on screen is always a link to its source: {register: id} or {file, line|match}.
function num(text, src) {
  if (text == null || text === "") text = "n/a";
  if (!src) return esc(text);
  return `<a class="num" data-src='${esc(JSON.stringify(src))}'>${esc(text)}</a>`;
}
const fmt = {
  q: (v) => (v == null ? "n/a" : v.toFixed(3)),
  ms: (v) => (v == null ? "n/a" : Math.round(v).toLocaleString() + " ms"),
  usd: (v) => (v == null ? "n/a" : "$" + v.toFixed(v < 1 ? 4 : 2)),
  int: (v) => (v == null ? "n/a" : Math.round(v).toLocaleString()),
};

async function openSource(src) {
  const dlg = $("#src-dialog");
  let title, body;
  try {
    if (src.register) {
      const row = await api(`/api/register?id=${encodeURIComponent(src.register)}`);
      title = `Register row ${src.register} (${row.status})`;
      body = JSON.stringify(row, null, 1);
    } else {
      const p = new URLSearchParams({ file: src.file });
      if (src.line != null) p.set("line", src.line);
      if (src.match) p.set("match", Object.entries(src.match).map(([k, v]) => `${k}:${v}`).join(","));
      const res = await api(`/api/source?${p}`);
      title = src.file + (src.line != null ? `, line ${src.line}` : "");
      if (src.note) title += ` (${src.note})`;
      let text = res.text;
      try { if (text) text = JSON.stringify(JSON.parse(text), null, 1); } catch (e) { /* raw */ }
      body = `sha256 at cache build: ${res.sha256_at_build || "n/a"}\n` +
        (res.match ? `rows matching ${JSON.stringify(res.match)}: ${res.n_matching_rows} (lines ${res.lines.join(", ")})\n` : "") +
        (text ? "\n" + text : "");
    }
  } catch (e) { title = "Source unavailable"; body = String(e); }
  $("#src-title").textContent = title;
  $("#src-body").textContent = body;
  dlg.showModal();
}
document.addEventListener("click", (ev) => {
  const a = ev.target.closest("a.num");
  if (a) { ev.preventDefault(); openSource(JSON.parse(a.dataset.src)); }
});

function inputs() {
  const f = $("#form");
  return {
    budget: +f.budget.value, floor: +f.floor.value, latency: +f.latency.value,
    hardware: [...document.querySelectorAll("#hw input:checked")].map((i) => i.value),
  };
}

// ── screen 1 ──
function renderMeta(m) {
  const b = [];
  if (m.backend.fake) b.push(`<div class="banner"><b><span class="badge fake">FAKE BACKEND</span></b>router: ${esc(m.backend.router)}; pareto: ${esc(m.backend.pareto)}. Frontier, recommendation and our side of the live run come from the dashboard's stand-in until src/dse lands on main.</div>`);
  b.push(`<div class="banner"><b>Offline snapshot</b>built ${esc(m.manifest.built_utc)} at commit ${esc(m.manifest.git_commit || "n/a")} from ${Object.keys(m.manifest.sources).length} committed files. Cloud rows are <span class="badge stub">STUB</span> (no key; src/cloud/client.py pricing table).</div>`);
  $("#banners").innerHTML = b.join("");
  const f = $("#form");
  f.budget.value = m.defaults.budget_usd; f.floor.value = m.defaults.quality_floor; f.latency.value = m.defaults.latency_target_ms;
  $("#hw").innerHTML = m.hardware.map((h) => {
    const on = h.machine && m.machines.includes(h.machine);
    return `<label class="${on ? "" : "off"}"><input type="checkbox" value="${esc(h.machine || h.name)}" ${on ? "checked" : "disabled"}> ${esc(h.name)}${h.machine ? ` (${esc(h.machine)})` : ""}${h.memory_gb ? `, ${num(h.memory_gb + " GB", { file: h.src })}` : ""}${on ? "" : " <span class='muted'>no measured envelope</span>"}</label>`;
  }).join("");
  $("#workload").innerHTML = m.workloads.map((w) => `<option value="${esc(w.id)}">${esc(w.label)}</option>`).join("");
  showWorkload();
}
function showWorkload() {
  const w = state.meta.workloads.find((x) => x.id === $("#workload").value);
  if (!w) return;
  $("#workload-desc").innerHTML = `${esc(w.description)} ... <span class="small">(${num(w.item_id, { file: w.src, match: { item_id: w.item_id } })}${w.scenario ? "; the live run replays " + esc(w.replay_src) : ""})</span>`;
}

// ── screen 2 ──
const RUNTIME_COLOR = { ollama: "var(--series-1)", llama_server: "var(--series-2)", cloud: "var(--series-3)" };
const runtimeOf = (p) => (p.runtime && p.runtime.startsWith("llama") ? "llama_server" : p.runtime);

function chart(machine, pts, inp, chosenId) {
  const W = 520, H = 300, L = 48, R = 16, T = 12, B = 40;
  const plotted = pts.filter((p) => p.quality != null && p.latency_p50_ms != null);
  const xmax = Math.max(inp.latency, ...plotted.map((p) => p.latency_p50_ms)) * 1.08 || 1;
  const x = (v) => L + (v / xmax) * (W - L - R), y = (v) => T + (1 - v) * (H - T - B);
  const step = xmax > 100000 ? 50000 : xmax > 40000 ? 20000 : 10000;
  let g = "";
  for (let q = 0; q <= 1.0001; q += 0.25) g += `<line class="gridl" x1="${L}" x2="${W - R}" y1="${y(q)}" y2="${y(q)}"/><text x="${L - 6}" y="${y(q) + 4}" text-anchor="end">${q.toFixed(2)}</text>`;
  for (let v = 0; v <= xmax; v += step) g += `<text x="${x(v)}" y="${H - B + 16}" text-anchor="middle">${v / 1000}s</text>`;
  g += `<line class="axisl" x1="${L}" x2="${W - R}" y1="${y(0)}" y2="${y(0)}"/>`;
  g += `<text x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">p50 latency</text>`;
  g += `<line class="floor" x1="${L}" x2="${W - R}" y1="${y(inp.floor)}" y2="${y(inp.floor)}"/><text class="lbl" x="${W - R}" y="${y(inp.floor) - 4}" text-anchor="end">floor ${inp.floor}</text>`;
  g += `<line class="target" x1="${x(inp.latency)}" x2="${x(inp.latency)}" y1="${T}" y2="${y(0)}"/><text class="lbl" x="${x(inp.latency) + 4}" y="${T + 10}">target</text>`;
  const fr = plotted.filter((p) => p.on_frontier).sort((a, b) => a.latency_p50_ms - b.latency_p50_ms);
  if (fr.length > 1) g += `<polyline class="front" points="${fr.map((p) => `${x(p.latency_p50_ms)},${y(p.quality)}`).join(" ")}"/>`;
  for (const p of plotted) {
    const cx = x(p.latency_p50_ms), cy = y(p.quality), col = RUNTIME_COLOR[runtimeOf(p)] || "var(--muted)";
    if (p.stub) g += `<circle class="dot" cx="${cx}" cy="${cy}" r="5" fill="var(--surface)" stroke="${col}" style="stroke:${col}"/><text class="lbl" x="${cx + 8}" y="${cy + 4}">STUB</text>`;
    else g += `<circle class="dot ${p.on_frontier ? "" : "dominated"}" cx="${cx}" cy="${cy}" r="5" fill="${col}"/>`;
    if ((p.flags || []).includes("bare template")) g += `<circle class="ring" cx="${cx}" cy="${cy}" r="9"/><text class="lbl" x="${cx + 12}" y="${cy - 6}">bare template</text>`;
    if (p.id === chosenId) g += `<path class="star" d="${starPath(cx, cy, 13, 6)}"/>`;
    g += `<circle class="hit" cx="${cx}" cy="${cy}" r="12" data-id="${esc(p.id)}"/>`;
  }
  return `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Quality vs latency, ${esc(machine)}">${g}</svg>`;
}
function starPath(cx, cy, ro, ri) {
  let d = "";
  for (let i = 0; i < 10; i++) {
    const r = i % 2 ? ri : ro, a = (Math.PI / 5) * i - Math.PI / 2;
    d += (i ? "L" : "M") + (cx + r * Math.cos(a)).toFixed(1) + "," + (cy + r * Math.sin(a)).toFixed(1);
  }
  return d + "Z";
}
function pointRow(p) {
  const badges = (p.stub ? '<span class="badge stub">STUB</span> ' : "") +
    ((p.flags || []).includes("bare template") ? '<span class="badge bare">bare template</span> ' : "") +
    (p.on_frontier ? '<span class="badge">frontier</span>' : "");
  return `<tr><td>${esc(p.model)}</td><td>${esc(p.config_id)}</td><td class="n">${num(fmt.q(p.quality), p.src)}</td><td class="n">${num(fmt.int(p.quality_n), p.src)}</td><td class="n">${num(fmt.ms(p.latency_p50_ms), p.src)}</td><td class="n">${num(fmt.ms(p.latency_p90_ms), p.src)}</td><td class="n">${num(fmt.usd(p.usd_per_1k_steps), p.src)}</td><td>${badges}</td></tr>`;
}
const TABLE_HEAD = "<tr><th>model</th><th>config</th><th>quality</th><th>n</th><th>p50</th><th>p90</th><th>USD / 1k steps</th><th></th></tr>";

function renderFrontier() {
  const fr = state.frontier, inp = inputs(), chosen = state.rec && state.rec.chosen ? state.rec.chosen.id : null;
  state.byId = {};
  $("#legend").innerHTML = [["ollama", "Ollama"], ["llama_server", "llama-server"]].map(([k, n]) =>
    `<span><svg width="10" height="10"><circle cx="5" cy="5" r="4" fill="${RUNTIME_COLOR[k]}"/></svg>${n}</span>`).join("") +
    `<span>line: frontier</span><span>star: recommended</span><span class="small">source files: ${fr.files.map((f) => num(f, { file: f })).join(", ")}</span>`;
  $("#charts").innerHTML = Object.entries(fr.machines).map(([m, v]) => {
    v.points.forEach((p) => (state.byId[p.id] = p));
    const rows = v.points.slice().sort((a, b) => (b.quality || 0) - (a.quality || 0)).map(pointRow).join("");
    return `<div class="chart"><h3>${esc(m)}</h3>${chart(m, v.points, inp, chosen)}<details><summary>Table view (${v.points.length} configs)</summary><div class="scroll"><table>${TABLE_HEAD}${rows}</table></div></details></div>`;
  }).join("") || "<p class='muted'>Select at least one measured hardware option.</p>";
  fr.cloud.forEach((p) => (state.byId[p.id] = p));
  $("#cloud").innerHTML = fr.cloud.length ? `<h3>Cloud <span class="badge stub">STUB</span></h3><p class="muted small">No cloud quality is measured yet; costs are the pricing table applied to the outcome rows' mean prompt size.</p><div class="scroll"><table>${TABLE_HEAD}${fr.cloud.map(pointRow).join("")}</table></div>` : "";
}
document.addEventListener("mousemove", (ev) => {
  const t = ev.target.closest && ev.target.closest("circle.hit"), tip = $("#tip");
  if (!t) { tip.hidden = true; return; }
  const p = state.byId[t.dataset.id];
  tip.innerHTML = `<b>${esc(p.model)}</b> ${esc(p.config_id)}<br>quality ${fmt.q(p.quality)} (n ${p.quality_n})<br>p50 ${fmt.ms(p.latency_p50_ms)}, p90 ${fmt.ms(p.latency_p90_ms)}<br>${fmt.usd(p.usd_per_1k_steps)} per 1k steps${p.stub ? "<br>STUB" : ""}${(p.flags || []).length ? "<br>" + esc(p.flags.join(", ")) : ""}<br><span class="muted">click for source</span>`;
  tip.hidden = false;
  tip.style.left = Math.min(ev.clientX + 14, innerWidth - 310) + "px";
  tip.style.top = ev.clientY + 14 + "px";
});
document.addEventListener("click", (ev) => {
  const t = ev.target.closest && ev.target.closest("circle.hit");
  if (t) openSource(state.byId[t.dataset.id].src);
});

// ── screen 3 ──
function renderRec() {
  const r = state.rec, c = r.chosen;
  const stub = '<span class="badge stub">STUB</span>';
  let h = "";
  if (c) {
    h += `<p class="hero">${esc(c.machine)} / ${esc(c.runtime)} / ${esc(c.model)} <span class="muted small">(${esc(c.config_id)})</span> ${(c.flags || []).includes("bare template") ? '<span class="badge bare">bare template</span>' : ""}</p>`;
    h += `<div class="rec">
      <div class="tile"><div class="k">Quality (n ${num(fmt.int(c.quality_n), c.src)})</div><div class="v">${num(fmt.q(c.quality), c.src)}</div></div>
      <div class="tile"><div class="k">p50 latency</div><div class="v">${num(fmt.ms(c.latency_p50_ms), c.src)}</div></div>
      <div class="tile"><div class="k">Cost per 1k steps</div><div class="v">${num(fmt.usd(c.usd_per_1k_steps), c.src)}</div></div>
      <div class="tile"><div class="k">Savings vs all-cloud per 1k steps ${stub}</div><div class="v">${num(fmt.usd(r.savings_vs_all_cloud_usd_per_1k), r.savings_src)}</div></div>
    </div>`;
  } else {
    h += `<p class="hero">No local config meets these inputs</p>`;
  }
  h += `<p>${esc(r.reason)}</p>`;
  if (r.alternatives.length) h += `<details open><summary>Next alternatives</summary><div class="scroll"><table>${TABLE_HEAD}${r.alternatives.map(pointRow).join("")}</table></div></details>`;
  h += `<p class="muted small">Backend: ${esc(r.backend)}. All-cloud comparison uses STUB cloud rows until real cloud rows exist.</p>`;
  $("#rec-body").innerHTML = h;
}

async function refresh() {
  const inp = inputs();
  const hw = encodeURIComponent(inp.hardware.join(","));
  try {
    [state.frontier, state.rec] = await Promise.all([
      api(`/api/frontier?hardware=${hw}`),
      api(`/api/recommend?budget=${inp.budget}&floor=${inp.floor}&latency=${inp.latency}&hardware=${hw}`),
    ]);
    renderRec(); renderFrontier();
  } catch (e) { $("#rec-body").textContent = "Error: " + e.message; }
}

// ── screen 4 ──
let scen = null;
function renderScenario(s) {
  scen = s;
  $("#session").innerHTML = s.sessions.map((x) => `<option value="${esc(x.key)}">${esc(x.model)}, seed ${x.seed}</option>`).join("");
  const reg = s.register;
  const cite = (id) => (reg[id] ? `${num(id, { register: id })}: ${esc(reg[id].value)}` : `${esc(id)}: not in register`);
  $("#evidence").innerHTML = `<p><b>Recorded outcome, all six sessions at 4096</b> (register, verbatim): ${cite("R2-real-v1-gated-kill")}</p>
    <p><b>Mechanism</b>: ${num("R2-mechanism-verdict", { register: "R2-mechanism-verdict" })}, ${num("R2-mechanism-lowlevel", { register: "R2-mechanism-lowlevel" })} (Ollama keeps the system message; llama.cpp context shift then discards the start of the prompt at 4096).</p>
    <p><b>Our side's quality</b>: <span class="badge proj">PROJECTED</span> ${esc(s.mitigation.label || "measured")}${s.mitigation.register_ids.length ? " (" + s.mitigation.register_ids.map((i) => num(i, { register: i })).join(", ") + ")" : ""}. Until ${esc(s.mitigation.file)} is synced and its register rows compute, the screen shows our router's decisions, not a measured outcome.</p>`;
  $("#live-intro").textContent = `Both sides replay the same recorded session from ${s.source} (arm ${s.arm}). The naive side's outcome per step is the recorded turn; our side shows what the router would send instead.`;
}
function meter(k, v) { return `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div></div>`; }
function stateIcon(cls, text) { return `<span class="st ${cls}">${esc(text)}</span>`; }

function addStep(s) {
  const n = s.naive, o = s.ours, d = o.decision, src = n.src;
  let out;
  if (n.outcome.recorded) {
    const oc = n.outcome, http = oc.http_status.join("/");
    const canary = oc.canary_check ? ` canary sys ${oc.canary_sys_ok ? "kept" : "LOST"}, hist ${oc.canary_hist_ok ? "kept" : "LOST"};` : "";
    out = `${stateIcon(oc.silent ? "bad" : "ok", oc.silent ? "silent failure" : "ok")} HTTP ${num(http, src)}, errors ${num(oc.errors.length, src)}; rules held ${num(oc.rules_held, src)};${canary} ${oc.failures.length ? "failed: " + esc(oc.failures.join(", ")) : ""}<br><span class="muted">transcript ${num(fmt.int(oc.transcript_tokens), src)} tokens vs loaded context ${num(fmt.int(oc.loaded_context), src)}</span>`;
  } else {
    out = `${stateIcon("warn", "cloud")} <span class="badge stub">STUB</span> ${esc(n.outcome.note)}`;
  }
  $("#naive-steps").insertAdjacentHTML("beforeend", `<div class="step"><div class="row1"><span class="turn">turn ${num(s.turn, src)}</span><b>${esc(n.target)}</b><span class="why">${esc(n.reason)}</span></div><div>${out}</div></div>`);
  const tk = d.tokens || {};
  const cost = d.stub ? `${num(fmt.usd(d.est_cost_usd), src)} <span class="badge stub">STUB</span>` : num(fmt.usd(d.est_cost_usd), src);
  $("#ours-steps").insertAdjacentHTML("beforeend", `<div class="step"><div class="row1"><span class="turn">turn ${num(s.turn, src)}</span><b>${esc(d.target)}</b><span class="why">${esc(d.reason)}</span></div><div>${stateIcon(o.system_prompt_in_request ? "ok" : "warn", o.system_prompt_in_request ? "rules in request" : "rules not sent locally")} num_ctx ${num(fmt.int(d.num_ctx), src)}; tokens system ${num(fmt.int(tk.system), src)} + history ${num(fmt.int(tk.kept_history), src)} + answer ${num(fmt.int(tk.answer_budget), src)} = ${num(fmt.int(tk.total), src)}; kept ${num(d.kept_turns, src)}, dropped ${num(d.dropped_turns, src)} turns; cost ${cost}; predicted latency ${num(fmt.ms(d.predicted_latency_ms), src)}</div></div>`);
  const rq = n.running_quality;
  $("#naive-meters").innerHTML = meter("Running cost", num(fmt.usd(n.running_cost_usd), src)) +
    (rq ? meter("Turns with every rule held", `${num(rq.turns_all_rules_held, src)} / ${num(rq.turns, src)}`) +
      meter("Canary checks passed", `${num(rq.canary_checks_passed, src)} / ${num(rq.canary_checks, src)}`) : "");
  $("#ours-meters").innerHTML = meter("Running cost" + (o.stub_cost ? ' <span class="badge stub">STUB</span>' : ""), num(fmt.usd(o.running_cost_usd), src)) +
    meter("Quality", o.quality.measured ? "measured, see register" : '<span class="badge proj">PROJECTED</span> <span class="small">pending mitigation run</span>');
  for (const id of ["#naive-steps", "#ours-steps"]) { const el = $(id); el.scrollTop = el.scrollHeight; }
}
function stopReplay() {
  if (state.es) { state.es.close(); state.es = null; }
  $("#play").disabled = false; $("#stop").disabled = true;
}
function play() {
  stopReplay();
  for (const id of ["#naive-steps", "#ours-steps", "#naive-meters", "#ours-meters"]) $(id).innerHTML = "";
  const inp = inputs();
  const p = new URLSearchParams({ session: $("#session").value, delay_ms: $("#delay").value, budget: inp.budget, floor: inp.floor, latency: inp.latency });
  const es = new EventSource(`/api/replay/stream?${p}`);
  state.es = es; $("#play").disabled = true; $("#stop").disabled = false;
  es.addEventListener("start", (e) => {
    const h = JSON.parse(e.data), np = h.naive_params;
    $("#naive-desc").textContent = `RouteLLM-style: difficulty = task words / ${np.difficulty_words}, cloud above ${np.difficulty_threshold}; length checked against the model's advertised context, never num_ctx.`;
    $("#ours-desc").textContent = `Router backend: ${h.router_backend}. Keeps the system prompt and the current turn, trims history to fit num_ctx, escalates when it cannot fit.`;
  });
  es.addEventListener("step", (e) => addStep(JSON.parse(e.data)));
  es.addEventListener("done", stopReplay);
  es.onerror = stopReplay;
}

async function main() {
  state.meta = await api("/api/meta");
  renderMeta(state.meta);
  renderScenario(await api("/api/scenario"));
  await refresh();
  $("#form").addEventListener("input", refresh);
  $("#hw").addEventListener("change", refresh);
  $("#workload").addEventListener("change", showWorkload);
  $("#use-example").addEventListener("click", () => {
    const w = state.meta.workloads.find((x) => x.id === $("#workload").value);
    showWorkload(); refresh();
    if (w && w.scenario) location.hash = "#live";
  });
  $("#play").addEventListener("click", play);
  $("#stop").addEventListener("click", stopReplay);
}
main().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p class="banner">Failed to load: ${esc(e.message)}</p>`); });
