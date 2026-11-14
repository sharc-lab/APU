"""Figure F7: per-turn event raster, one row per window tier (llama3.1:8b, one session per tier), plus the NVIDIA
4096 session as an extra row next to the AMD 4096 row.

AMD rows. Register row R2-mechanism-lowlevel (numbers_register.compute_r2_mechanism_lowlevel; result file
results/x2_r2_mechanism.jsonl). The script reads the same `r2m_call` records that function reads, grouped by
`arm_id` the same way, and marks per turn (`turn_idx`) whether any call of that turn had a message-level truncation
(`message_level_truncation`), a llama.cpp context shift (`context_shifts` non-empty), or an Ollama token-level cut
(`token_level_cut`). Before drawing it checks its per-tier totals against the row's own value string.

NVIDIA row. Register row blade-mechanism-4096 (numbers_register.compute_blade_mechanism_4096) lists only the session
report results/blade_r2_mechanism_4096_v2.jsonl.report.json, which has totals but no per-turn events. The per-turn
events come from the raw file that report was built from, results/blade_r2_mechanism_4096_v2.jsonl (same
`r2m_call` record format as the AMD file). The script refuses to draw unless that file's totals equal the VERIFIED
report's: calls, message-truncated calls, first truncated turn, context-shift events and token-level-cut calls.

Usage: py -3.12 analysis/figures/fig_f7.py   (writes docs/paper1/figures/fig_f7.pdf)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import COLORS, REPO, entry, plt, row, save, style  # noqa: E402

ROWS = ["R2-mechanism-lowlevel", "blade-mechanism-4096"]
NAME = "fig_f7.pdf"
NVIDIA_RAW = "results/blade_r2_mechanism_4096_v2.jsonl"
EVENTS = [("message drop", "s", COLORS[0]), ("context shift", "D", COLORS[1]), ("token cut", "v", COLORS[6])]


def _calls(path):
    rows = [json.loads(x) for x in (REPO / path).read_text(encoding="utf-8").splitlines() if x.strip()]
    by = {}
    for r in rows:
        if r.get("record") == "r2m_call":
            by.setdefault(r["arm_id"].replace("_call2_notools", ""), []).append(r)
    return by


def _per_turn(rs):
    ev = {k: set() for k, _, _ in EVENTS}
    for r in rs:
        if r.get("message_level_truncation"):
            ev["message drop"].add(r["turn_idx"])
        if r.get("context_shifts"):
            ev["context shift"].add(r["turn_idx"])
        if r.get("token_level_cut"):
            ev["token cut"].add(r["turn_idx"])
    return ev


def load():
    e = entry(ROWS[0])
    reg = row(ROWS[0])["value"]
    amd = _calls(e["data_files"][0])
    for arm, rs in amd.items():
        m = re.search(re.escape(arm) + r": (\d+)/(\d+) calls message-truncated; (\d+) context shifts", reg)
        n_cs = sum(len(r.get("context_shifts") or []) for r in rs)
        n_mt = sum(1 for r in rs if r.get("message_level_truncation"))
        if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != (n_mt, len(rs), n_cs):
            raise RuntimeError(f"{arm}: per-call totals do not match {ROWS[0]}")
    entry(ROWS[1])
    rep = json.loads((REPO / entry(ROWS[1])["data_files"][0]).read_text(encoding="utf-8"))["sessions"][0]
    nv = _calls(NVIDIA_RAW)
    if len(nv) != 1:
        raise RuntimeError(f"expected one arm in {NVIDIA_RAW}")
    (nv_arm, nv_rs), = nv.items()
    got = dict(n_calls=len(nv_rs),
               n_calls_message_truncated=sum(1 for r in nv_rs if r.get("message_level_truncation")),
               first_truncated_turn=min(r["turn_idx"] for r in nv_rs if r.get("message_level_truncation")),
               context_shift_events=sum(len(r.get("context_shifts") or []) for r in nv_rs),
               token_level_cut_calls=sum(1 for r in nv_rs if r.get("token_level_cut")))
    want = {k: rep[k] for k in got}
    if got != want:
        raise RuntimeError(f"{NVIDIA_RAW} totals {got} differ from the verified report {want}")
    return amd, nv_arm, nv_rs


def tier_label(arm):
    return "default" if arm == "ollama_default" else arm.replace("ollama_ctx_", "")


def draw(out_dir=None):
    style()
    amd, nv_arm, nv_rs = load()
    rows = [(f"AMD {tier_label(a)}", _per_turn(rs), rs) for a, rs in amd.items()]
    rows.append((f"NVIDIA {tier_label(nv_arm)}", _per_turn(nv_rs), nv_rs))
    turns = sorted({r["turn_idx"] for _, _, rs in rows for r in rs})
    fig, ax = plt.subplots(figsize=(3.45, 1.95))
    n = len(rows)
    for i, (lab, ev, _) in enumerate(rows):
        y = n - 1 - i
        if i == n - 1:
            ax.axhspan(y - 0.5, y + 0.5, color="#f2f2f2", zorder=0)
        for k, (name, mk, col) in enumerate(EVENTS):
            xs = sorted(ev[name])
            ax.plot(xs, [y + (1 - k) * 0.25] * len(xs), mk, ms=2.6 if mk != "v" else 3.6, color=col, mec="white", mew=0.3, ls="",
                    zorder=3)
    ax.set_yticks(range(n))
    ax.set_yticklabels([lab for lab, _, _ in rows][::-1], fontsize=5.8)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(min(turns) - 0.8, max(turns) + 0.8)
    ax.set_xlabel("turn")
    ax.grid(axis="y", visible=False)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker=mk, color=col, ls="", ms=3, label=nm) for nm, mk, col in EVENTS],
              loc="upper center", bbox_to_anchor=(0.45, 1.16), ncol=3, frameon=False, columnspacing=1.0,
              handletextpad=0.2)
    return save(fig, NAME, out_dir)


def main(out_dir=None):
    return draw(out_dir)


if __name__ == "__main__":
    print(main())
