// The four demo views (A to D). Uses $, esc, num, api, fmt from app.js (loaded first). Plain JS + inline SVG.
"use strict";
const LABEL_CLASS = { "measured": "measured", "projected": "projected", "simulated": "simulated",
  "priced, not executed": "priced", "not measured": "notmeasured" };
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
  let h = `<p class="hero">${r.status === "not measured" ? lab("not measured") + " " : ""}${esc(r.basis)}</p>`;
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

async function viewsMain() {
  try { renderSilent(await api("/api/silent")); } catch (e) { $("#silent-body").textContent = "Error: " + e.message; }
  try {
    renderDeviceOptions(await api("/api/device"));
    $("#devform").addEventListener("input", refreshDevice);
    $("#devform").addEventListener("change", refreshDevice);
    await refreshDevice();
  } catch (e) { $("#device-body").textContent = "Error: " + e.message; }
}
viewsMain();
