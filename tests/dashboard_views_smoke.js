// Node smoke test for demo/dashboard/static/views.js (run by tests/test_dashboard_views.py when node is installed).
// Stubs the DOM and the API with JSON the dashboard's own App produced, runs every renderer and the story-mode
// navigation, prints one JSON line.
// usage: node dashboard_views_smoke.js <views.js> <api_dump.json>
const fs = require("fs");
const dump = JSON.parse(fs.readFileSync(process.argv[3], "utf8"));
const els = {};
const listeners = {};
function el(sel) {
  if (!els[sel]) {
    const o = { sel, innerHTML: "", textContent: "", disabled: false, value: "0.05", parentNode: null, children: [],
      addEventListener(t, f) { (listeners[sel + ":" + t] = listeners[sel + ":" + t] || []).push(f); },
      insertAdjacentHTML() {}, appendChild(c) { c.parentNode = o; o.children.push(c.sel); },
      classList: { s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); },
        toggle(c, on) { if (on === undefined ? !this.s.has(c) : on) this.s.add(c); else this.s.delete(c); },
        contains(c) { return this.s.has(c); } },
      device: { innerHTML: "", value: "evo-x2" }, model: { innerHTML: "", value: "qwen3:14b" },
      workload: { innerHTML: "", value: "r2_agent" }, maxtimeout: { value: "0.05" } };
    els[sel] = o;
  }
  return els[sel];
}
global.document = { querySelector: el, body: el("body"), addEventListener() {} };
global.location = { search: "", hash: "" };
global.$ = el;
global.esc = (s) => String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
global.num = (t, src) => (t == null || t === "" ? "n/a" : (src ? `<a>${esc(t)}</a>` : esc(t)));
global.api = async (p) => { const k = p.split("?")[0]; if (!dump[k]) throw new Error("no dump " + k); return dump[k]; };
global.fetch = async () => ({ json: async () => dump.POST });
const text = (h) => h.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
eval(fs.readFileSync(process.argv[2], "utf8") + "\n;global.__r = { renderEscalationResult, storyNav, storyGo, story };");
setTimeout(() => {
  const out = {};
  for (const k of ["#silent-body", "#device-body", "#cost-body", "#esc-task"]) out[k] = text(el(k).innerHTML || el(k).textContent);
  const R = global.__r;
  const walk = [];
  const snap = () => ({ title: el("#story-title").textContent, indicator: el("#story-indicator").textContent,
    back_disabled: el("#story-back").disabled, next: el("#story-next").textContent,
    story_on: el("body").classList.contains("story-on"), panel: el("#story-panel").children.slice(-1)[0] || null,
    items: text(el("#story-items").innerHTML) });
  walk.push(snap());
  for (let i = 0; i < 4; i++) { R.storyGo("next"); walk.push(snap()); }
  R.storyGo("back"); walk.push(snap());
  R.storyGo("next"); R.storyGo("next"); walk.push(snap());  // past the last step: free exploration
  out.story = walk;
  out.nav = [R.storyNav(0, 5, "back"), R.storyNav(0, 5, "next"), R.storyNav(4, 5, "next"), R.storyNav(2, 5, 9),
             R.storyNav(3, 5, "explore")];
  R.renderEscalationResult(dump.POST);
  out["#esc-result"] = text(el("#esc-result").innerHTML);
  console.log(JSON.stringify(out));
}, 300);
