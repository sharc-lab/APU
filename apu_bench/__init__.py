"""apu_bench: an open benchmark harness and a "bring your own logs" audit for local and hybrid agents.

Two commands (py -3.12 -m apu_bench --help):
  run    a device profile plus a B3 v2 task subset -> result JSON (versioned, hardware-tagged, reproducible) and a
         readable Markdown report: device, runtime, windows, wrong rate, silent vs flagged, seconds per task.
         --runtime fake builds the full report from fake records (offline, testable); --runtime live composes the
         x2_r2_evalset command and runs it only with --execute (never run so far; a real run needs a
         pre-registration, docs/APU_BENCH_RFP_MAPPING.md).
  audit  replay recorded agent sessions against a device profile: hidden failure rate (the runtime's own truncation
         or context-shift log lines, detector v2) and cloud cost (billed when the log carries usage, else a
         caching floor and a no-caching upper bound). docs/BYOL_AUDIT_DESIGN.md.

Formats are versioned by the constants below; a reader refuses a schema it does not know.
"""
BENCH_VERSION = "0.1.0"
PROFILE_SCHEMA = "apu-device-profile/1"
RESULT_SCHEMA = "apu-bench-result/1"
SESSION_SCHEMA = "apu-audit-session/1"
AUDIT_SCHEMA = "apu-audit-result/1"
