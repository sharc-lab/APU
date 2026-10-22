// Node smoke test for demo/dashboard/static/views.js (run by tests/test_dashboard_views.py when node is installed).
// Stubs the DOM and the API with JSON the dashboard's own App produced, runs every renderer, prints one JSON line.
// usage: node dashboard_views_smoke.js <views.js> <api_dump.json>
const fs = require("fs");
const dump = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const els = {};
function el(sel) {
  if (!els[sel]) els[sel] = { innerHTML: "", textContent: "", disabled: false, value: "0.05",
    addEventListener() {}, insertAdjacentHTML() {},
    device: { innerHTML: "", value: "evo-x2" }, model: { innerHTML: "", value: "qwen3:14b" },
    workload: { innerHTML: "", value: "r2_agent" }, maxtimeout: { value: "0.05" } };
  return els[sel];
}
global.document = { querySelector: el };
global.$ = el;
global.esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
global.num = (t, src) => (t == null || t === "" ? "n/a" : (src ? `<a>${esc(t)}</a>` : esc(t)));
global.api = async (p) => { const k = p.split("?")[0]; if (!dump[k]) throw new Error("no dump " + k); return dump[k]; };
global.fetch = async () => ({ json: async () => dump.POST });
const text = (h) => h.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
eval(fs.readFileSync(process.argv[2], "utf8") + "\n;global.__r = { renderEscalationResult };");
setTimeout(() => {
  const out = {};
  for (const k of ["#silent-body", "#device-body", "#cost-body", "#esc-task"]) out[k] = text(el(k).innerHTML || el(k).textContent);
  global.__r.renderEscalationResult(dump.POST);
  out["#esc-result"] = text(el("#esc-result").innerHTML);
  console.log(JSON.stringify(out));
}, 300);
