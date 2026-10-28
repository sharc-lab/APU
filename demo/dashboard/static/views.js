// The four demo views (A to D). Uses $, esc, num, api, fmt from app.js (loaded first). Plain JS + inline SVG.
"use strict";
const LABEL_CLASS = { "measured": "measured", "projected": "projected", "simulated": "simulated",
  "priced, not executed": "priced", "not measured": "notmeasured", "pending flash-attention control": "simulated", "being measured": "notmeasured" };
function lab(l) { return l ? `<span class="lab ${LABEL_CLASS[l] || ""}">${esc(l)}</span>` : ""; }
const R = (id) => ({ register: id });
function survival(intact, n, checkpoints) {
  return `<span class="surv" title="sessions intact at turns ${checkpoints.join(", ")}">${intact.map((k, i) =>
    `<i class="${k === n ? "on" : k > 0 ? "part" : ""}" title="turn ${checkpoints[i]}: ${k}/${n} intact"></i>`).join("")}</span>`;
}
const tierName = (t) => t.replace("ollama_", "").replace("num_ctx_", "ctx ").replace("ctx_", "ctx ").replace("_numkeep", " + num_keep");

// ── A: silent failure ──
function renderSilent(s) {
  const cp = s.v1b_per_model.checkpoints;
  const card = (dev, title) => {
    const d = s.device_pair[dev], ev = d.evidence, rid = d.register_id;
    if (!d.parsed) return `<div class="devcard"><h3>${esc(title)}</h3>${lab("not measured")} register row ${num(rid, R(rid))} not parsed</div>`;
    const anyAfter = d.models.some((m) => m.silent_after_window > 0);
    return `<div class="devcard ${anyAfter ? "bad" : "ok"}"><h3>${esc(title)} ${lab(d.label)}</h3>
      ${d.models.map((m) => `<p><b>${esc(m.model)}</b>: Ollama loaded ${num(m.loaded_window.join(", "), R(rid))} tokens;
        first turn over the window ${num(m.first_over_window_turn.join(", ") || "never", R(rid))}<br>
        <span class="big">${num(m.gated_silent + "/" + m.n, R(rid))}</span> sessions silently lost a rule,
        ${num(m.silent_after_window, R(rid))} after the window filled ${survival(m.intact, m.n, cp)}</p>`).join("")}</div>`;
  };
  const wr = s.wrong_rate;
  let bars = `${lab("not measured")} register row ${num(wr.register_id, R(wr.register_id))} not parsed`;
  if (wr.parsed) {
    const W = 640, rowH = 30, L = 170, H = wr.tiers.length * rowH + 24, max = Math.max(50, ...wr.tiers.map((t) => t.pct));
    const x = (v) => L + (v / max) * (W - L - 150);
    let g = "";
    wr.tiers.forEach((t, i) => {
      const y = 8 + i * rowH;
      g += `<text x="${L - 8}" y="${y + 15}" text-anchor="end">${esc(tierName(t.tier))} (${t.loaded_context.join(",")})</text>`;
      g += `<rect class="bar ${t.tier === "ollama_default" ? "base" : ""}" x="${L}" y="${y + 3}" width="${Math.max(1, x(t.pct) - L)}" height="${rowH - 10}" rx="2"/>`;
      g += `<text x="${x(t.pct) + 6}" y="${y + 15}">${t.pct.toFixed(1)}% wrong, ${t.surfaced} surfaced</text>`;
    });
    bars = `<svg class="bars" viewBox="0 0 ${W} ${H}" role="img" aria-label="qwen3:14b wrong-task rate by window">${g}</svg>
      <div class="scroll"><table><tr><th>window (loaded)</th><th>wrong tasks</th><th>silent</th><th>surfaced</th><th>excess over default, points (95% CI, session level)</th><th>file check</th></tr>
      ${wr.tiers.map((t) => { const fc = wr.file_check[t.tier] || {}; return `<tr><td>${esc(tierName(t.tier))} (${num(t.loaded_context.join(","), R(wr.register_id))})</td>
        <td class="n">${num(`${t.wrong}/${t.n} (${t.pct.toFixed(1)}%)`, R(wr.register_id))}</td><td class="n">${num(t.silent, R(wr.register_id))}</td>
        <td class="n">${num(t.surfaced, R(wr.register_id))}</td>
        <td class="n">${t.excess_points == null ? "baseline" : num(`${t.excess_points >= 0 ? "+" : ""}${t.excess_points.toFixed(1)} (${t.ci_lo.toFixed(1)} to ${t.ci_hi.toFixed(1)})`, R(wr.register_id))}</td>
        <td class="small">${fc.wrong_rows != null ? num(`${fc.wrong_rows} wrong rows, ${fc.http_error_rows} with an HTTP error`, { file: wr.file, line: fc.first_line }) : "n/a"}</td></tr>`; }).join("")}</table></div>`;
  }
  const v = s.v1b_per_model;
  let grid = `${lab("not measured")} register row ${num(v.register_id, R(v.register_id))} not parsed`;
  if (v.parsed) {
    const models = [...new Set(v.cells.map((c) => c.model))], tiers = [...new Set(v.cells.map((c) => c.tier))];
    grid = `<div class="scroll"><table><tr><th>window</th>${models.map((m) => `<th>${esc(m)}</th>`).join("")}</tr>${tiers.map((t) =>
      `<tr><td>${esc(tierName(t))}</td>${models.map((m) => { const c = v.cells.find((x) => x.tier === t && x.model === m);
        return c ? `<td>${num(c.gated_silent + "/" + c.n, R(v.register_id))} silent, ${num(c.silent_after_window, R(v.register_id))} after window ${survival(c.intact, c.n, cp)}</td>` : "<td>n/a</td>"; }).join("")}</tr>`).join("")}</table></div>`;
  }
  $("#silent-body").innerHTML = `
    <p class="muted">${esc(s.device_pair.caption)}</p>
    <div class="pair">${card("cpu", "evo-t2s, Intel CPU: Ollama default")}${card("igpu", "evo-t2s, Intel iGPU: Ollama default")}</div>
    <h3 style="margin-top:16px">qwen3:14b on evo-x2: wrong task results by window ${lab(wr.label)}</h3>
    <p class="muted small">30 task turns per session, 5 sessions per window, graded by end state. Every wrong result came back as a normal HTTP 200 answer. Source: ${num(wr.register_id, R(wr.register_id))}; CIs: ${num(wr.ci_register_id, R(wr.ci_register_id))}.</p>
    ${bars}
    <details><summary>Rule loss per model and window on evo-x2, 5 seeds (strengthened run v1 + v1b) ${lab(v.label)}</summary>
      <p class="muted small">Gated: rules and metrics the model holds at 90% or more of its own validation baseline. Boxes: sessions with every gated rule intact at turns ${cp.join(", ")}. Source: ${num(v.register_id, R(v.register_id))}.</p>${grid}</details>`;
}

// ── B: per-device recommendation ──
const SETTING_NAMES = [["window", "Context window (tokens)"], ["num_keep", "num_keep"], ["kv_precision", "KV cache precision"],
  ["flash_attention", "Flash attention"], ["context_handling", "Runtime and context handling"]];
function renderDeviceOptions(o) {
  const f = $("#devform");
  f.device.innerHTML = o.devices.map((d) => `<option value="${esc(d.id)}">${esc(d.label)}</option>`).join("");
  f.model.innerHTML = o.models.map((m) => `<option value="${esc(m)}"${m === "qwen3:14b" ? " selected" : ""}>${esc(m)}</option>`).join("");
  f.workload.innerHTML = o.workloads.map((w) => `<option value="${esc(w.id)}">${esc(w.label)}</option>`).join("");
  f.maxtimeout.value = o.default_max_timeout_share;
}
function renderDevice(r) {
  const tile = ([k, name]) => {
    const s = r.settings[k] || {};
    return `<div class="tile"><div class="k">${esc(name)} ${lab(s.label)}</div><div class="v small" style="font-size:16px">${num(s.value, s.src)}</div>${s.note ? `<div class="muted small">${esc(s.note)}</div>` : ""}</div>`;
  };
  const d = r.decision || {};
  let h = d.text ? `<p class="lead">${num(d.text, d.src)} ${lab(d.label)}${(d.tolerances || []).length
    ? `<br><span class="small">Would route local only at tolerance &gt;= ${d.tolerances.map((t) => `${num(t.tolerance_local.toFixed(1) + " points", d.src)} vs ${esc(t.model)} (local ${esc(t.local)} vs all-cloud ${esc(t.cloud)})`).join("; ")}; ${esc(d.tolerances[0].basis)}.</span>`
    : d.route_cost ? `<br><span class="small">Route at start: ${num(d.route_cost.text, d.route_cost.src)} ${lab(d.route_cost.label)}</span>` : ""}</p>` : "";
  h += `<p class="hero">${r.status !== "measured" ? lab(r.status) + " " : ""}${esc(r.basis)}</p>`;
  h += `<div class="settings">${SETTING_NAMES.map(tile).join("")}</div>`;
  if (r.outcome.length) h += `<h3>Measured outcome</h3><ul>${r.outcome.map((o) => `<li>${lab(o.label)} ${num(o.text, o.src)}</li>`).join("")}</ul>`;
  if (r.excluded.length) h += `<h3>Excluded by the usability floor (timeout share above ${esc(r.max_timeout_share)})</h3><ul>${r.excluded.map((e) => `<li>${esc(e.config_id)}: ${num(e.reason, e.src)}</li>`).join("")}</ul>`;
  if (r.caveats.length) h += `<p class="muted small">${r.caveats.map(esc).join("<br>")}</p>`;
  if (r.alternatives.length) h += `<details><summary>Other measured options (${r.alternatives.length})</summary><pre class="answer">${esc(r.alternatives.map((a) => JSON.stringify(a)).join("\n"))}</pre></details>`;
  h += `<p class="muted small">Register rows: ${r.register_ids.map((i) => num(i + (r.register[i] ? ` (${r.register[i].status})` : ""), R(i))).join(", ") || "none"}.</p>`;
  $("#device-body").innerHTML = h;
}
async function refreshDevice() {
  const f = $("#devform");
  const p = new URLSearchParams({ device: f.device.value, model: f.model.value, workload: f.workload.value, max_timeout_share: f.maxtimeout.value });
  try { renderDevice(await api(`/api/device/recommend?${p}`)); } catch (e) { $("#device-body").textContent = "Error: " + e.message; }
}

// ── C: cost panel ──
function renderCost(c) {
  let h = "";
  if (c.lead) {
    const L = c.lead;
    h += `<p class="lead">${num(L.text, L.src)} ${lab(L.label)}</p>${L.note ? `<p class="muted small">${esc(L.note)}</p>` : ""}`;
  }
  h += `<ul class="costlist">${c.items.map((i) => `<li><span class="what">${esc(i.what)}</span> ${lab(i.label)}<br>${num(i.text, i.src)}</li>`).join("")}
    <li><span class="what">Router savings</span><br>${num(c.router.text, c.router.src)}</li></ul>`;
  h += `<h3 style="margin-top:16px">Cost documents</h3>` + c.docs.map((d) => d.present
    ? `<div class="docbox"><b>${num(d.path, { file: d.path, line: d.headline_line })}</b>${d.title ? ": " + esc(d.title) : ""}<br>
       <span class="small">Quoted verbatim (each claim carries its own label in the doc):</span> ${esc(d.headline || "(no paragraph after the title)")}
       ${d.sections.length ? `<details><summary>${d.sections.length} sections</summary><ul class="small">${d.sections.map((s) => `<li>${num(s.text, { file: d.path, line: s.line })}</li>`).join("")}</ul></details>` : ""}</div>`
    : `<div class="docbox"><b>${esc(d.path)}</b>: not present yet. This panel picks it up when the file lands; no restart needed.</div>`).join("");
  $("#cost-body").innerHTML = h;
}

// ── D: live escalation ──
function escStatus(s) {
  return `live calls ${s.live_requested ? "on" : "off"}; key ${s.key_present ? "present" : "absent"}; cap ${num("USD " + s.cap_usd.toFixed(2), null)}; ` +
    `demo ledger ${esc(s.ledger)}: ${num("USD " + s.ledger_total_usd.toFixed(4), null)} spent; live calls this run ${s.live_calls_used}/${s.live_calls_allowed}`;
}
function renderEscalationTask(e) {
  const t = e.task;
  if (!t.available) { $("#esc-task").innerHTML = `${lab("not measured")} ${esc(t.reason)}`; $("#esc-run").disabled = true; return; }
  const loc = t.local, pr = t.redo || {};
  $("#esc-task").innerHTML = `
    <p><b>The task</b> (${esc(t.task_id)}, ${esc(t.local_model)} at ${esc(tierName(t.tier))}, seed ${t.seed}, turn ${t.turn_idx}; ${esc(t.selection)}):</p>
    <div class="answer">${esc(t.user_message)}</div>
    <div class="pair" style="margin-top:8px">
      <div class="devcard bad"><h3>Local answer ${lab("measured")}</h3>
        <p class="small">HTTP error: ${num(loc.any_http_error ? "yes" : "no", loc.src)}; graded end state: ${num(loc.end_state_ok ? "correct" : "wrong", loc.src)}; window ${num(loc.loaded_context, loc.src)}</p>
        <div class="answer">${esc(loc.tool_calls.length ? JSON.stringify(loc.tool_calls) : "(no tool call)")}\n${esc(loc.final_text)}</div>
        <p class="small muted">Expected: ${esc(JSON.stringify(t.expected_calls))}</p></div>
      <div class="devcard"><h3>Cloud (${esc(t.cloud_model)}), recorded full-history redo of this task ${lab(pr.label)}</h3>
        ${pr.usd != null
          ? `<p><span class="big">${num("USD " + pr.usd.toFixed(4), pr.src)}</span>${pr.seconds != null ? ` in ${num(pr.seconds.toFixed(1) + " s", pr.src)}` : ""}</p>${pr.cloud_final_text ? `<div class="answer">${esc(pr.cloud_final_text)}</div>` : ""}`
          : `<p class="lead">${esc(pr.text || "Cost to fix a wrong answer: being measured")}</p>`}
        <p class="small muted">${esc(pr.note || "")} Click "Send to the cloud" for one live call through the capped client.</p></div>
    </div>`;
  $("#esc-mode").innerHTML = escStatus(e.status);
}
function renderEscalationResult(r) {
  const live = r.mode === "executed";
  $("#esc-result").innerHTML = `<div class="devcard">
    <h3>${live ? "Cloud answer, live now" : r.usd != null ? "Not sent live: recorded full-history redo of this task" : "Not sent live"} ${lab(r.label)}</h3>
    <p class="small">${esc(r.reason)}</p>
    <div class="settings">
      <div class="tile"><div class="k">USD</div><div class="v">${r.usd == null ? "being measured" : num("USD " + r.usd.toFixed(4), live ? null : r.usd_src)}</div><div class="muted small">${esc(r.usd_basis || "")}</div></div>
      <div class="tile"><div class="k">Seconds</div><div class="v">${r.seconds == null ? "being measured" : r.seconds.toFixed(1) + " s"}</div></div>
    </div>
    ${live ? `<div class="answer">${esc((r.tool_calls || []).length ? JSON.stringify(r.tool_calls) : "(no tool call)")}\n${esc(r.answer || "")}</div>`
      : r.answer ? `<div class="answer">${esc(r.answer)}</div>` : ""}
    <p class="muted small">${escStatus(r)}</p></div>`;
  $("#esc-mode").innerHTML = escStatus(r);
}
async function runEscalation() {
  $("#esc-run").disabled = true;
  $("#esc-result").innerHTML = `<p class="muted">sending</p>`;
  try {
    const r = await fetch("/api/escalate", { method: "POST" });
    renderEscalationResult(await r.json());
  } catch (e) { $("#esc-result").textContent = "Error: " + e.message; }
  $("#esc-run").disabled = false;
}

// ── story mode: five guided steps, Next/Back, then free exploration ──
// Pure navigation (tested under node): index in [0, n-1]; "next" on the last step leaves story mode.
function storyNav(index, n, action) {
  if (action === "next") return index >= n - 1 ? { index, explore: true } : { index: index + 1, explore: false };
  if (action === "back") return { index: Math.max(0, index - 1), explore: false };
  if (action === "explore") return { index, explore: true };
  if (typeof action === "number") return { index: Math.max(0, Math.min(n - 1, action)), explore: false };
  return { index, explore: false };
}
const STORY_PANELS = { silent: "#silent-body", device: "#device-wrap", escalate: "#esc-wrap" };
const story = { data: null, index: 0, on: false, tol: 5, machine: "AMD workstation" };
function returnPanels() {
  for (const [k, sel] of Object.entries(STORY_PANELS)) {
    const node = $(sel), home = document.querySelector(`.home[data-home="${k}"]`);
    if (node && home && node.parentNode !== home) home.appendChild(node);
  }
}
function setStoryMode(on) {
  story.on = on;
  document.body.classList.toggle("story-on", on);
  if (!on) returnPanels();
}
// Step 4 trade-off: options as rows (savings per month at the stated volume, accuracy, silent wrong answers); the
// ceiling is drawn only as a dotted line with its label, never as a row.
function tradeoffHTML(t) {
  if (!t || !t.featured) return `<p class="muted small">${lab("not measured")} trade-off row not parsed</p>`;
  const v = String(t.volume);
  const block = (r, title, label) => {
    const opts = [["All-cloud", r.all_cloud], ["Naive local at the device default", r.naive_local], ["Hybrid, window-overflow flag", r.hybrid_flag]];
    const vals = opts.map(([, o]) => o.saves[v]).concat([r.ceiling.saves[v], 0]);
    const lo = Math.min(...vals), hi = Math.max(...vals), W = 560, L = 230, H = opts.length * 26 + 30;
    const x = (u) => L + ((u - lo) / ((hi - lo) || 1)) * (W - L - 20);
    let g = `<line class="ci" x1="${x(0)}" x2="${x(0)}" y1="4" y2="${H - 22}"/>`;
    opts.forEach(([name, o], i) => {
      const y = 8 + i * 26, x0 = x(Math.min(0, o.saves[v])), x1 = x(Math.max(0, o.saves[v]));
      g += `<text x="${L - 8}" y="${y + 13}" text-anchor="end">${esc(name)}</text><rect class="bar${o.saves[v] < 0 ? " base" : ""}" x="${x0}" y="${y + 2}" width="${Math.max(1, x1 - x0)}" height="16" rx="2"/>`;
    });
    const cx = x(r.ceiling.saves[v]);
    g += `<line x1="${cx}" x2="${cx}" y1="4" y2="${H - 22}" stroke="currentColor" stroke-dasharray="4 4"/><text x="${cx}" y="${H - 8}" text-anchor="middle">ceiling, no such detector</text>`;
    const rows = opts.map(([name, o]) => `<tr><td>${esc(name)}</td><td class="n">${num("USD " + o.saves[v].toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }), t.src)}</td><td class="n">${num(`${o.correct}/${o.n} (${o.pct.toFixed(1)}%)`, t.src)}</td><td class="n">${o.silent == null ? "0" : num(o.silent, t.src)}${o.escalated != null ? `; escalated ${num(o.escalated, t.src)}` : ""}</td></tr>`).join("");
    return `<h3>${esc(title)} ${lab(label)}</h3><svg class="bars" viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(title)}">${g}</svg>
      <div class="scroll"><table><tr><th>option</th><th>saves per month at ${Number(v).toLocaleString()} tasks/day</th><th>correct</th><th>wrong answers silent</th></tr>${rows}</table></div>
      <p class="muted small">Side note, not an option: a detector for in-window wrong answers would save up to ${num("USD " + r.ceiling.saves[v].toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }), t.src)} at ${num(`${r.ceiling.correct}/${r.ceiling.n}`, t.src)} (ceiling, no such detector).</p>`;
  };
  const f = t.featured;
  let h = block(f, `${f.device} at its default vs ${f.model}, B3 tasks`, f.device_label);
  if (t.cpu) h += block(t.cpu, `${t.cpu.device} at 4096 vs ${t.cpu.model}, B3 tasks`, t.cpu.device_label);
  return h;
}
// Step 3 accuracy-tolerance slider. routeAt mirrors views.route_at: local if tolerance >= the entry's threshold.
function routeAt(e, tol) {
  const local = tol + 1e-9 >= e.tolerance_local;
  return { route: local ? "local" : "cloud", saves: local ? (e.saves_local ? e.saves_local["1000"] : null) : 0 };
}
function routeRows(r, tol) {
  const usd = (v) => (v == null ? "n/a" : "USD " + v.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 }));
  return r.entries.map((e) => {
    const d = routeAt(e, tol);
    return `<tr><td>${esc(e.device)}<br><span class="muted small">${esc(e.setting)}</span></td><td>${esc(e.model)}</td>
      <td class="n">${num(`${e.local} (${e.local_pct.toFixed(1)}%)`, r.src)}${e.r2_basis ? `<br><span class="small">${esc(r.basis_label)}</span>` : ""}</td>
      <td class="n">${num(`${e.cloud} (${e.cloud_pct.toFixed(1)}%)`, r.src)}</td><td class="n">${num(e.gap.toFixed(1) + " points", r.src)}</td>
      <td><b>${d.route === "local" ? "local, recommended setting" : "cloud from the start"}</b></td><td class="n">${num(usd(d.saves), r.src)}</td></tr>`;
  }).join("");
}
function routeHTML(r, tol) {
  if (!r || !r.entries || !r.entries.length) return `<p>${lab("being measured")} routing at session start</p>`;
  return `<div class="controls"><label>Accuracy gap you accept (points)<input type="range" id="route-tol" min="${r.min}" max="${r.max}" step="0.1" value="${tol}"></label>
      <span class="big" id="route-tol-v">${tol.toFixed(1)}</span> ${lab("measured")}</div>
    <div class="scroll"><table id="route-table"><tr><th>local option</th><th>cloud tier</th><th>local accuracy</th><th>cloud accuracy</th><th>gap</th><th>decision at this tolerance</th><th>saves per month at ${r.volume.toLocaleString()} tasks/day</th></tr>${routeRows(r, tol)}</table></div>
    <p class="muted small">${esc(r.not_measured)} Savings when routed local are the whole all-cloud bill of that tier; 0 when the session goes to the cloud.</p>`;
}
function bindRoute(r) {
  const s = $("#route-tol");
  if (!s || !s.addEventListener) return;
  s.addEventListener("input", () => {
    const t = +s.value;
    $("#route-tol-v").textContent = t.toFixed(1);
    $("#route-table").innerHTML = `<tr><th>local option</th><th>cloud tier</th><th>local accuracy</th><th>cloud accuracy</th><th>gap</th><th>decision at this tolerance</th><th>saves per month at ${r.volume.toLocaleString()} tasks/day</th></tr>` + routeRows(r, t);
  });
}
function renderStory() {
  const steps = story.data.steps, st = steps[story.index], n = steps.length;
  $("#story-steps").innerHTML = steps.map((s, i) => `<li class="${i === story.index ? "cur" : i < story.index ? "done" : ""}"><button type="button" data-step="${i}">${s.n}</button></li>`).join("");
  $("#story-title").textContent = st.title;
  $("#story-indicator").textContent = `Step ${st.n} of ${n}`;
  $("#story-back").disabled = story.index === 0;
  $("#story-next").textContent = story.index === n - 1 ? "Finish" : "Next";
  returnPanels();
  $("#story-items").innerHTML = "";
  $("#story-panel").innerHTML = storyMainHTML(st) + storyDetailsHTML(st);
  if (st.kind === "money") bindTolerance(st);
  if (st.kind === "tool") bindTool(st);
}
// ── plain main screens (no links, ids or paths); evidence in a closed Details panel per step ──
const TAG = (t) => (t ? `<span class="tag-s">${esc(t)}</span>` : "");
function storyMainHTML(st) {
  const m = st.main || {};
  if (st.kind === "cards") return `<div class="s-cards">${(m.cards || []).map((c) => `<div class="s-card ${c.tone || ""}"><div class="s-big">${esc(c.big)}</div><p>${esc(c.text)} ${TAG(c.tag)}</p></div>`).join("")}</div>`;
  if (st.kind === "bars") {
    const max = Math.max(50, ...(m.bars || []).map((b) => b.value));
    return `<div class="s-bars">${(m.bars || []).map((b) => `<div class="s-bar-row"><span class="s-bar-label">${esc(b.label)}</span><span class="s-bar" style="width:${(b.value / max) * 60}%"></span><span class="s-big s-bar-v">${b.value}%</span> ${TAG(b.tag)}</div>`).join("")}</div>${m.caption ? `<p class="s-caption">${esc(m.caption)} ${TAG(m.caption_tag)}</p>` : ""}`;
  }
  if (st.kind === "tiles") return `<div class="s-tiles">${(m.tiles || []).map((t) => `<div class="s-tile">${t.big ? `<div class="s-big">${esc(t.big)}</div>` : ""}<p>${esc(t.text)} ${TAG(t.tag)}</p></div>`).join("")}</div>`;
  if (st.kind === "money") {
    const sl = m.slider;
    return `<div class="s-cards">${(m.numbers || []).map((c, i) => `<div class="s-card"><div class="s-big">${esc(c.big)}</div><p>${esc(c.text)} ${TAG(c.tag)}</p>${i === 1 ? `<button type="button" id="s-live" class="s-live">${esc(m.live_button)}</button><div id="s-live-out"></div>` : ""}</div>`).join("")}</div>
      <p class="s-take">${esc(m.takeaway)}</p>
      <div class="s-slider"><label>Accuracy gap you accept <input type="range" id="s-tol" min="${sl.min}" max="${sl.max}" step="0.1" value="${story.tol}"> <b id="s-tol-v">${story.tol.toFixed(1)}</b> points</label>
      <div id="s-tol-out">${toleranceLines(sl, story.tol)}</div></div>`;
  }
  if (st.kind === "tool") {
    return `<div class="s-pick"><label>Machine <select id="s-machine">${m.machines.map((x) => `<option${x === story.machine ? " selected" : ""}>${esc(x)}</option>`).join("")}</select></label></div><div id="s-tool-card">${toolCard(st)}</div>`;
  }
  return "";
}
function tierEntries(entries, machine, tier) { return entries.filter((e) => e.machine === machine && e.model === tier); }
function toleranceLines(sl, tol) {
  const seen = {};
  return sl.entries.filter((e) => e.model === "claude-sonnet-5-5").map((e) => {
    if (seen[e.machine]) return ""; seen[e.machine] = 1;
    const d = routeAt(e, tol);
    // the decision is measured; a dollar figure is measured per-task cost x an assumed volume, so it is projected
    return `<p class="s-line">${esc(e.machine)}: ${d.route === "local" ? `runs locally ${TAG("measured")}, saves $${Math.round(d.saves).toLocaleString()} a month at ${sl.volume.toLocaleString()} tasks a day ${TAG("projected")}` : `goes to the cloud ${TAG("measured")}`}</p>`;
  }).join("");
}
function liveHTML(r) {
  const cents = r.usd == null ? null : (r.usd * 100).toFixed(1);
  const tag = r.label === "measured" ? "measured" : "recorded";
  const head = r.mode === "executed" ? "Live fix" : "Recorded fix of the same answer";
  let ans = r.answer ? String(r.answer) : "";
  try { const o = JSON.parse(ans); if (o && typeof o.answer === "string") ans = o.answer; } catch (e) { /* plain text */ }
  if (ans.length > 220) ans = ans.slice(0, 217) + "...";
  return `<p class="s-line"><b>${head}:</b> ${cents == null ? "not available" : cents + " cents"}${r.seconds != null ? ", " + r.seconds.toFixed(1) + " s" : ""} ${TAG(tag)}</p>${ans ? `<p class="s-answer">"${esc(ans)}"</p>` : ""}`;
}
function bindTolerance(st) {
  const s = $("#s-tol");
  if (s && s.addEventListener) s.addEventListener("input", () => { story.tol = +s.value; $("#s-tol-v").textContent = story.tol.toFixed(1); $("#s-tol-out").innerHTML = toleranceLines(st.main.slider, story.tol); });
  const b = $("#s-live");
  if (b && b.addEventListener) b.addEventListener("click", async () => {
    b.disabled = true;
    $("#s-live-out").innerHTML = `<p class="s-line">sending</p>`;
    try { const r = await fetch("/api/fix-now", { method: "POST" }); story.live = await r.json(); $("#s-live-out").innerHTML = liveHTML(story.live); }
    catch (e) { $("#s-live-out").textContent = "Error: " + e.message; }
    b.disabled = false;
  });
  if (story.live && $("#s-live-out")) $("#s-live-out").innerHTML = liveHTML(story.live);
}
function toolCard(st) {
  const m = st.main, e = tierEntries(m.entries, story.machine, m.default_tier)[0];
  if (!e) return `<div class="s-card"><div class="s-big">not measured</div><p>No measurements for this machine yet. ${TAG("not measured")}</p></div>`;
  const d = routeAt(e, story.tol), local = d.route === "local";
  const acc = local ? e.local_pct : e.cloud_pct;
  return `<div class="s-card ${local ? "good" : ""}"><div class="s-head">${local ? "Run locally with this setting" : "Send to the cloud from the start"}</div>
    <p>Expected accuracy <span class="s-big">${Math.round(acc)}%</span> ${TAG(local && e.r2_basis ? "measured (session pass rate)" : "measured")}</p>
    <p>Saves per month at ${m.volume.toLocaleString()} tasks a day <span class="s-big">$${Math.round(d.saves || 0).toLocaleString()}</span> ${TAG("projected")}</p>
    <p class="s-note">At an accuracy gap of ${story.tol.toFixed(1)} points (step 4).</p></div>`;
}
function bindTool(st) {
  const s = $("#s-machine");
  if (!s || !s.addEventListener) return;
  s.addEventListener("change", () => { story.machine = s.value; $("#s-tool-card").innerHTML = toolCard(st); });
}
function storyDetailsHTML(st) {
  const items = (st.items || []).map((i) => `<li>${lab(i.label)} ${i.src ? num(i.text, i.src) : esc(i.text)}</li>`).join("");
  const extra = st.tradeoff ? tradeoffHTML(st.tradeoff) : "";
  return `<details class="s-details"><summary>Details: the measurements behind this step</summary><ul class="small">${items}</ul>${extra}</details>`;
}
function storyGo(action) {
  const r = storyNav(story.index, story.data.steps.length, action);
  story.index = r.index;
  if (r.explore) { setStoryMode(false); location.hash = "#silent"; return; }
  renderStory();
}
async function initStory() {
  story.data = await api("/api/story");
  $("#story-next").addEventListener("click", () => storyGo("next"));
  $("#story-back").addEventListener("click", () => storyGo("back"));
  $("#story-steps").addEventListener("click", (ev) => { const b = ev.target.closest && ev.target.closest("button[data-step]"); if (b) storyGo(+b.dataset.step); });
  $("#story-open").addEventListener("click", (ev) => { ev.preventDefault(); setStoryMode(true); storyGo(0); });
  $("#eng-link").addEventListener("click", (ev) => { ev.preventDefault(); storyGo("explore"); });
  document.addEventListener("keydown", (ev) => {
    if (!story.on || /INPUT|SELECT|TEXTAREA/.test((ev.target && ev.target.tagName) || "")) return;
    if (ev.key === "ArrowRight") storyGo("next");
    if (ev.key === "ArrowLeft") storyGo("back");
  });
  const params = new URLSearchParams(location.search);
  const s4 = story.data.steps.find((s) => s.kind === "money");
  story.tol = s4 ? s4.main.slider.default : 5;
  const t = parseFloat(params.get("tol"));
  if (!isNaN(t)) story.tol = t;
  if (params.get("machine")) story.machine = params.get("machine");
  // story mode is the landing page; ?explore=1 (or ?view=engineering) opens the engineering view
  if (params.get("explore") !== "1" && params.get("view") !== "engineering") {
    setStoryMode(true); storyGo(+(params.get("step") || 1) - 1);
  }
}

async function viewsMain() {
  try { renderSilent(await api("/api/silent")); } catch (e) { $("#silent-body").textContent = "Error: " + e.message; }
  try {
    renderDeviceOptions(await api("/api/device"));
    $("#devform").addEventListener("input", refreshDevice);
    $("#devform").addEventListener("change", refreshDevice);
    await refreshDevice();
  } catch (e) { $("#device-body").textContent = "Error: " + e.message; }
  try { renderCost(await api("/api/cost")); } catch (e) { $("#cost-body").textContent = "Error: " + e.message; }
  try {
    renderEscalationTask(await api("/api/escalation"));
    $("#esc-run").addEventListener("click", runEscalation);
  } catch (e) { $("#esc-task").textContent = "Error: " + e.message; }
  try { await initStory(); } catch (e) { $("#story-title").textContent = "Story mode unavailable: " + e.message; setStoryMode(false); }
}
viewsMain();
