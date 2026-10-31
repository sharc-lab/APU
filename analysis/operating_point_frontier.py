"""Accuracy vs cloud USD per task frontier for the named operating point q4_0 28928 (operator 2026-10-30).

Every plotted value is parsed from the register (docs/NUMBERS_REGISTER.md), not recomputed:
  - headline points: row operating-point-q4-28928 (q4_0 28928, f16 8192, claude-sonnet-5-5 all-cloud);
  - de-emphasized points: row hybrid-full-pipeline-v1 (the other configs, and naive local q4_0 28928 at USD 0).
Writes docs/figures/operating_point_frontier.png and docs/figures/operating_point_frontier.json. All values are
replay of recorded runs (row labels carried into the figure). Regenerate the register first.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTER = "docs/NUMBERS_REGISTER.md"
OUT_PNG = "docs/figures/operating_point_frontier.png"
OUT_JSON = "docs/figures/operating_point_frontier.json"
OP_ROW = "operating-point-q4-28928"
HFP_ROW = "hybrid-full-pipeline-v1"

_POINT = re.compile(r"(?P<name>q4_0 28928|q8_0 15360|f16 8192|f16 4096)(?: \([^)]*\))?(?: ceiling \d+)?: kept local "
                    r"(?P<kept>\d+)/(?P<n>\d+), accuracy (?P<c>\d+)/(?P=n) \((?P<acc>[\d.]+)%\), cloud USD "
                    r"(?P<usd>[\d.]+)/task")
_ALLC_OP = re.compile(r"claude-sonnet-5-5 all-cloud: accuracy (?P<c>\d+)/(?P<n>\d+) \((?P<acc>[\d.]+)%\), cloud USD "
                      r"(?P<usd>[\d.]+)/task")
_NAIVE = re.compile(r"naive local (?P<c>\d+)/(?P<n>\d+)")


def row_value(repo: Path, claim: str) -> str:
    for ln in (Path(repo) / REGISTER).read_text(encoding="utf-8").splitlines():
        if ln.startswith(f"| {claim} |"):
            cols = ln.split(" | ")
            if cols[1] != "VERIFIED":
                raise ValueError(f"row {claim} is {cols[1]}")
            return cols[2]
    raise KeyError(claim)


def segments(v: str) -> list:
    return v.split(" / ")


def frontier_data(repo: Path = REPO) -> dict:
    op = row_value(repo, OP_ROW)
    hfp = row_value(repo, HFP_ROW)
    pts = []
    head = {}
    for seg in segments(op):
        for m in _POINT.finditer(seg):
            head[m["name"]] = m
        a = _ALLC_OP.search(seg)
        if a:
            allc = a
    for name in ("q4_0 28928", "f16 8192"):
        m = head[name]
        pts.append({"label": f"hybrid {name}", "accuracy": float(m["acc"]), "usd_task": float(m["usd"]),
                    "kept_local": int(m["kept"]), "n": int(m["n"]), "role": "headline", "row": OP_ROW})
    pts.append({"label": "claude-sonnet-5-5 all-cloud", "accuracy": float(allc["acc"]), "usd_task": float(allc["usd"]),
                "kept_local": 0, "n": int(allc["n"]), "role": "headline", "row": OP_ROW})
    for seg in segments(hfp):
        m = _POINT.search(seg)
        if not m:
            continue
        if m["name"] not in head:
            pts.append({"label": f"hybrid {m['name']}", "accuracy": float(m["acc"]), "usd_task": float(m["usd"]),
                        "kept_local": int(m["kept"]), "n": int(m["n"]), "role": "other config", "row": HFP_ROW})
        if m["name"] == "q4_0 28928":
            nv = _NAIVE.search(seg)
            pts.append({"label": "naive local q4_0 28928", "accuracy": round(100 * int(nv["c"]) / int(nv["n"]), 1),
                        "usd_task": 0.0, "kept_local": int(nv["n"]), "n": int(nv["n"]), "role": "other config",
                        "row": HFP_ROW})
    return {"label": "replay of recorded runs (B3 v2, qwen3:14b local on evo-x2, claude-sonnet-5-5, seeds "
                     "20260901-03); escalations of q4_0 28928 imputed (row operating-point-q4-28928)",
            "x": "cloud USD per task", "y": "accuracy (%)", "points": pts}


def plot(d: dict, out_png: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for p in d["points"]:
        head = p["role"] == "headline"
        ax.scatter(p["usd_task"], p["accuracy"], s=70 if head else 30, color="#1f4e79" if head else "#b0b0b0",
                   zorder=3 if head else 2)
        ax.annotate(f"{p['label']}\n{p['accuracy']:.1f}%, USD {p['usd_task']:.5f}"
                    + (f", {p['kept_local']}/{p['n']} local" if p["kept_local"] else ""),
                    (p["usd_task"], p["accuracy"]), textcoords="offset points", xytext=(6, -12 if head else 4),
                    fontsize=7 if head else 6, color="#1f1f1f" if head else "#808080")
    hp = sorted((p for p in d["points"] if p["role"] == "headline"), key=lambda p: p["usd_task"])
    ax.plot([p["usd_task"] for p in hp], [p["accuracy"] for p in hp], color="#1f4e79", lw=1, zorder=1)
    ax.axhline(97, color="#c00000", lw=0.8, ls="--")
    ax.text(0.0, 97.2, "pre-registered 97%", fontsize=6, color="#c00000")
    ax.set_xlabel(d["x"])
    ax.set_ylabel(d["y"])
    ax.set_title("Accuracy vs cloud cost per task (replay of recorded runs)", fontsize=9)
    ax.set_xlim(-0.0005, 0.0155)
    ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, d["label"], fontsize=5.5, color="#505050")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main(repo: Path = REPO) -> dict:
    d = frontier_data(repo)
    (Path(repo) / OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    (Path(repo) / OUT_JSON).write_text(json.dumps(d, indent=1) + "\n", encoding="utf-8")
    plot(d, Path(repo) / OUT_PNG)
    return d


if __name__ == "__main__":
    for p in main()["points"]:
        print(p)
