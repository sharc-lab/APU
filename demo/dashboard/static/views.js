// The four demo views (A to D). Uses $, esc, num, api, fmt from app.js (loaded first). Plain JS + inline SVG.
"use strict";
const LABEL_CLASS = { "measured": "measured", "projected": "projected", "simulated": "simulated",
  "priced, not executed": "priced", "not measured": "notmeasured", "pending flash-attention control": "simulated" };
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
    const first = ev.sessions[0];
    return `<div class="devcard ${anyAfter ? "bad" : "ok"}"><h3>${esc(title)} ${lab(d.label)}</h3>
      ${d.models.map((m) => `<p><b>${esc(m.model)}</b>: Ollama loaded ${num(m.loaded_window.join(", "), R(rid))} tokens;
        first turn over the window ${num(m.first_over_window_turn.join(", ") || "never", R(rid))}<br>
        <span class="big">${num(m.gated_silent + "/" + m.n, R(rid))}</span> sessions silently lost a rule,
        ${num(m.silent_after_window, R(rid))} after the window filled ${survival(m.intact, m.n, cp)}</p>`).join("")}
      <p class="small">Calls that surfaced an error to the client: ${num(ev.totals.calls_surfaced_error + " of " + ev.totals.calls, first ? first.src : { file: ev.file })}
        (HTTP status other than 200 or an error field, counted over every call in ${esc(ev.file)}).</p></div>`;
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
  let h = `<p class="hero">${r.status !== "measured" ? lab(r.status) + " " : ""}${esc(r.basis)}</p>`;
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
    h += `<p class="lead">${num(L.text, L.src)} ${lab(L.label)}</p>
      <p class="muted small">Every wrong task of the qwen3:14b task run (${num(L.n, L.src)} tasks) re-run on ${esc(L.model)}: mean ${num("USD " + L.usd_mean.toFixed(4), L.src)}, median ${num("USD " + L.usd_median.toFixed(4), L.src)}${L.seconds_mean != null ? `, mean ${num(L.seconds_mean.toFixed(2) + " s", L.src)} per task` : ""}. ${L.executed
        ? "Executed on the cloud; this panel shows the cost only."
        : "Priced from the local run's own token counts; seconds were not measured in that run."}</p>`;
  } else {
    h += `<p class="lead">${lab("not measured")} cloud redo cost per wrong task: register row not parsed</p>`;
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
  const loc = t.local, pr = t.redo, ran = pr.mode === "executed";
  $("#esc-task").innerHTML = `
    <p><b>The task</b> (${esc(t.task_id)}, ${esc(t.local_model)} at ${esc(tierName(t.tier))}, seed ${t.seed}, turn ${t.turn_idx}; ${esc(t.selection)}):</p>
    <div class="answer">${esc(t.user_message)}</div>
    <div class="pair" style="margin-top:8px">
      <div class="devcard bad"><h3>Local answer ${lab("measured")}</h3>
        <p class="small">HTTP error: ${num(loc.any_http_error ? "yes" : "no", loc.src)}; graded end state: ${num(loc.end_state_ok ? "correct" : "wrong", loc.src)}; window ${num(loc.loaded_context, loc.src)}</p>
        <div class="answer">${esc(loc.tool_calls.length ? JSON.stringify(loc.tool_calls) : "(no tool call)")}\n${esc(loc.final_text)}</div>
        <p class="small muted">Expected: ${esc(JSON.stringify(t.expected_calls))}</p></div>
      <div class="devcard"><h3>Cloud (${esc(t.cloud_model)}), recorded redo of this task ${lab(pr.label)}</h3>
        <p><span class="big">${num("USD " + pr.usd.toFixed(4), pr.src)}</span>${ran && pr.seconds != null ? ` in ${num(pr.seconds.toFixed(1) + " s", pr.src)}` : ""}<br>
        <span class="small">${ran ? `executed in the cloud redo run, ${num(pr.input_tokens.toLocaleString(), pr.src)} input tokens, prompt caching priced (all input uncached: ${num("USD " + pr.usd_all_uncached.toFixed(4), pr.src)})`
          : `priced from the local run's own ${num(pr.input_tokens.toLocaleString(), pr.src)} input tokens with caching (all input uncached: ${num("USD " + pr.usd_all_uncached.toFixed(4), pr.src)})`}; register ${num(pr.register_id, R(pr.register_id))}</span></p>
        ${ran && pr.cloud_final_text ? `<div class="answer">${esc(pr.cloud_final_text)}</div>` : ""}</div>
    </div>`;
  $("#esc-mode").innerHTML = escStatus(e.status);
}
function renderEscalationResult(r) {
  const live = r.mode === "executed";
  $("#esc-result").innerHTML = `<div class="devcard">
    <h3>${live ? "Cloud answer, live now" : "Not sent live: recorded redo of this task"} ${lab(r.label)}</h3>
    <p class="small">${esc(r.reason)}</p>
    <div class="settings">
      <div class="tile"><div class="k">USD</div><div class="v">${r.usd == null ? "n/a" : num("USD " + r.usd.toFixed(4), live ? null : r.usd_src)}</div><div class="muted small">${esc(r.usd_basis || "")}</div></div>
      <div class="tile"><div class="k">Seconds</div><div class="v">${r.seconds == null ? "not measured" : r.seconds.toFixed(1) + " s"}</div></div>
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
}
viewsMain();
