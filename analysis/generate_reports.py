#!/usr/bin/env python3
"""
generate_reports.py

Reads claude_code_characterization.json (and optionally Zachary's
replication_remote_search_v3.json) and produces:
  - reports/results_summary.md
  - reports/results_summary.pdf

Run from the SHARC root:
    python analysis/generate_reports.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT        = Path(__file__).parent.parent
ADAPTER_OUT = ROOT / "results" / "claude_code_characterization.json"
ZACHARY_OUT = ROOT / "results" / "zachary" / "replication_remote_search_v3.json"
MD_OUT      = ROOT / "reports" / "results_summary.md"
PDF_OUT     = ROOT / "reports" / "results_summary.pdf"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def fmt(v, decimals=2):
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def stats_row(label: str, s: dict) -> str:
    return (
        f"| {label} | {fmt(s['median'])} | {fmt(s['q1'])} | "
        f"{fmt(s['q3'])} | {fmt(s['min'])} | {fmt(s['max'])} | {int(s['n'])} |"
    )


# ---------------------------------------------------------------------------
# Markdown builder
# ---------------------------------------------------------------------------

def build_md(adapter: dict, zachary: dict | None) -> str:
    lines = []
    a = adapter

    agg   = a["aggregate"]
    env   = a["env"]
    cfg   = a["config"]
    seeds = a["per_seed_artifacts"]

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    lines += [
        "# SHARC — APU Characterization Results",
        "",
        f"> Generated: {now}",
        "",
        "## 1. Experiment Overview",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Experiment | {a['experiment']} |",
        f"| Result validity | `{a['result_validity']}` |",
        f"| Backend | {cfg['backend']} |",
        f"| Model | {seeds[0]['config'].get('mode', 'N/A')} |",
        f"| Seeds | {cfg['seeds']} |",
        f"| Sessions per seed | {cfg['sessions']} |",
        f"| Total sessions | {len(cfg['seeds']) * cfg['sessions']} |",
        f"| Instr version | {cfg['instr_version']} |",
        f"| Generated | {a['generated_utc']} |",
        "",
        "## 2. Environment",
        "",
        f"| Field | Value |",
        f"|---|---|",
        f"| Python | {env['python']} |",
        f"| Platform | {env['platform']} |",
        f"| CPU | {env['cpu_model']} |",
        f"| Cores (logical) | {env['cores_logical']} |",
        f"| Cores (physical) | {env['cores_physical']} |",
        f"| RAM | {env['ram_gb']} GB |",
        "",
        "## 3. Aggregate CPU Statistics",
        "",
        "All timing values in **milliseconds** unless noted. Stats are across seeds (n=5).",
        "",
        "| Metric | Median | Q1 | Q3 | Min | Max | N |",
        "|---|---|---|---|---|---|---|",
        stats_row("batch_host_cpu_ms",         agg["batch_host_cpu_ms"]),
        "",
        "### 3.1 CPU Category Breakdown (% of instrumented CPU, pooled across seeds)",
        "",
        "| Category | Median % | Q1 % | Q3 % | Min % | Max % |",
        "|---|---|---|---|---|---|",
    ]

    pct_fields = [
        ("TOOL_COMPUTE",        "pooled_tool_compute_pct"),
        ("FRAMEWORK",           "pooled_framework_pct"),
        ("ORCH (setup+disp)",   "pooled_orch_pct"),
        ("CLIENT_HTTP",         "pooled_client_http_pct"),
        ("CLIENT_PARSE",        "pooled_client_parse_pct"),
        ("HARNESS strict",      "pooled_harness_strict_pct"),
        ("HARNESS broad",       "pooled_harness_broad_pct"),
        ("RESIDUAL",            "pooled_residual_unattributed_pct"),
    ]
    for label, key in pct_fields:
        if key in agg:
            s = agg[key]
            lines.append(
                f"| {label} | {fmt(s['median'])}% | {fmt(s['q1'])}% | "
                f"{fmt(s['q3'])}% | {fmt(s['min'])}% | {fmt(s['max'])}% |"
            )

    lines += [
        "",
        "### 3.2 Per-Task Host CPU (ms, across seeds)",
        "",
        "| Task | Category | Median ms | Q1 | Q3 | Min | Max | N |",
        "|---|---|---|---|---|---|---|---|",
    ]

    TASK_CATS = {
        "CH-01": "code+hybrid", "CH-02": "code+hybrid",
        "CN-01": "compute-num", "FO-01": "file/output",
        "LH-01": "long-horizon", "LH-02": "long-horizon",
        "RE-01": "retrieval", "RE-02": "retrieval",
        "RH-01": "ret+hybrid", "RH-02": "ret+hybrid",
        "SH-01": "search+hybrid", "SH-02": "search+hybrid",
        "SO-01": "search-only", "SW-01": "sweep/canary",
    }

    for task_id, s in sorted(agg["per_task_host_cpu_ms"].items()):
        cat = TASK_CATS.get(task_id, "")
        lines.append(
            f"| {task_id} | {cat} | {fmt(s['median'])} | {fmt(s['q1'])} | "
            f"{fmt(s['q3'])} | {fmt(s['min'])} | {fmt(s['max'])} | {int(s['n'])} |"
        )

    lines += [
        "",
        "## 4. Per-Seed Summary",
        "",
        "| Seed | Sessions | Batch wall (s) | Host CPU (ms) | Residual % |",
        "|---|---|---|---|---|",
    ]

    for a_seed in seeds:
        run = a_seed["run"]
        cpu_ms = run["totals"]["instrumented_cpu_ns"] / 1e6
        res_pct = run["residual_fraction"] * 100
        lines.append(
            f"| {a_seed['config']['seed']} | {a_seed['config']['sessions']} | "
            f"{fmt(a_seed['batch_wall_s'])} | {fmt(cpu_ms)} | {fmt(res_pct)}% |"
        )

    # Zachary comparison if available
    if zachary:
        za = zachary["aggregate"]
        lines += [
            "",
            "## 5. Comparison with Zachary's LangGraph Baseline",
            "",
            "| Metric | This Run (OpenAI/gpt-4o-mini) | Zachary (OpenAI/LangGraph) |",
            "|---|---|---|",
        ]
        def cmp_row(label, our_key, zach_key=None):
            zach_key = zach_key or our_key
            our = agg.get(our_key, {})
            zch = za.get(zach_key, {})
            our_med = fmt(our.get("median", 0))
            zch_med = fmt(zch.get("median", 0))
            return f"| {label} | {our_med} | {zch_med} |"

        lines += [
            cmp_row("batch_host_cpu_ms",    "batch_host_cpu_ms"),
            cmp_row("TOOL_COMPUTE %",       "pooled_tool_compute_pct"),
            cmp_row("FRAMEWORK %",          "pooled_framework_pct"),
            cmp_row("ORCH %",               "pooled_orch_pct"),
            cmp_row("HARNESS strict %",     "pooled_harness_strict_pct"),
            cmp_row("RESIDUAL %",           "pooled_residual_unattributed_pct"),
        ]

        lines += [
            "",
            "> **Zachary's setup:** LangGraph + OpenAI, Linux/WSL2, Intel Core Ultra 5 325, "
            "5 seeds × 10 sessions, remote search.",
            "> **This run:** Direct OpenAI SDK, gpt-4o-mini, mock tools, same seed/session count.",
        ]

    lines += [
        "",
        "## 6. Harness Definitions",
        "",
        f"- **Strict:** `{agg['harness_strict_definition']}`",
        f"- **Broad:** `{agg['harness_broad_definition']}`",
        "",
        "## 7. Validity",
        "",
        f"- `result_validity`: **{a['result_validity']}**",
        f"- Publishable criteria: n≥5 seeds, residual_fraction < 15% per seed",
        "",
        "---",
        "_SHARC APU Characterization Project — auto-generated by analysis/generate_reports.py_",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

def _safe(text: str, maxlen: int = 60) -> str:
    """Strip markdown punctuation and truncate for PDF cells."""
    clean = (
        text.replace("**", "").replace("`", "").replace("*", "")
        .replace("\u2014", "--").replace("\u2013", "-")
        .replace("\u2022", "*").replace("\u2265", ">=").replace("\u2264", "<=")
        .replace("\u00d7", "x").replace("\u00b1", "+/-")
    )
    # Strip any remaining non-latin-1 characters
    clean = clean.encode("latin-1", errors="replace").decode("latin-1")
    return clean[:maxlen]


def build_pdf(md_text: str, out_path: Path) -> None:
    try:
        from fpdf import FPDF
    except ImportError:
        print("fpdf2 not installed -- skipping PDF. Run: pip install fpdf2")
        return

    PAGE_W = 170  # usable width in mm (A4 minus 20mm margins each side)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_margins(20, 20, 20)

    def reset_x():
        pdf.set_x(pdf.l_margin)

    for raw_line in md_text.split("\n"):
        line = raw_line.strip()
        reset_x()

        if line.startswith("# "):
            pdf.set_font("Helvetica", "B", 16)
            pdf.set_text_color(20, 20, 80)
            pdf.multi_cell(PAGE_W, 9, _safe(line[2:], 120))
            pdf.ln(2)
            pdf.set_text_color(0, 0, 0)

        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 12)
            pdf.set_text_color(40, 40, 120)
            pdf.multi_cell(PAGE_W, 7, _safe(line[3:], 120))
            pdf.ln(1)
            pdf.set_text_color(0, 0, 0)

        elif line.startswith("### "):
            pdf.set_font("Helvetica", "B", 10)
            pdf.set_text_color(60, 60, 150)
            pdf.multi_cell(PAGE_W, 6, _safe(line[4:], 120))
            pdf.set_text_color(0, 0, 0)

        elif line.startswith("|"):
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if not cells or all(set(c) <= set("-: ") for c in cells):
                continue  # separator row
            reset_x()
            pdf.set_font("Courier", "", 7)
            n = max(len(cells), 1)
            col_w = PAGE_W / n
            max_chars = max(int(col_w / 1.8), 8)
            for cell in cells:
                pdf.cell(col_w, 5, _safe(cell, max_chars), border=1)
            pdf.ln()

        elif line.startswith("> "):
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_text_color(80, 80, 80)
            pdf.multi_cell(PAGE_W, 5, _safe(line[2:], 200))
            pdf.set_text_color(0, 0, 0)

        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(PAGE_W, 5, "  - " + _safe(line[2:], 180))

        elif line == "---":
            pdf.ln(3)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(20, pdf.get_y(), 190, pdf.get_y())
            pdf.ln(3)

        elif line == "":
            pdf.ln(3)

        else:
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(PAGE_W, 5, _safe(line, 300))

    pdf.output(str(out_path))
    print(f"PDF written to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    adapter = load(ADAPTER_OUT)
    if not adapter:
        # Try root dir for backwards compat
        alt = ROOT / "claude_code_characterization.json"
        adapter = load(alt)
        if adapter:
            # Copy to results/
            import shutil
            shutil.copy(alt, ADAPTER_OUT)

    if not adapter:
        print(f"ERROR: could not find adapter output at {ADAPTER_OUT}")
        sys.exit(1)

    zachary = load(ZACHARY_OUT)
    if zachary:
        print(f"Zachary baseline loaded from {ZACHARY_OUT}")
    else:
        print("Zachary baseline not found — skipping comparison section")

    md_text = build_md(adapter, zachary)
    MD_OUT.write_text(md_text, encoding="utf-8")
    print(f"Markdown written to {MD_OUT}")

    build_pdf(md_text, PDF_OUT)


if __name__ == "__main__":
    main()
