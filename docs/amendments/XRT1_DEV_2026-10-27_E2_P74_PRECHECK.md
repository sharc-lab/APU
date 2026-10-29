# XRT-1 deviation: E2 at 74% produced no data (port pre-check bug) (Tue Oct 27 2026)

Recorded Tue Oct 27 2026, afternoon ET, by Zach's instruction, before the E2-at-74% re-run.
Registration: docs/registrations/BR_XRT_2026-10-26.md (unchanged). Predictions: unchanged.

## What happened (window_20261027_130904_xrt1_e2_p74, EXPECT_HEAD 6f1209e)
- All four E2 cells at 74% (sycl f16, sycl q4_0, vulkan f16, vulkan q4_0) stopped before their first
  trial with `XRT_ERROR RuntimeError: port 8091 already answers /health (status -1)`.
- Status -1 is the tool's code for a client timeout, not an HTTP answer. tools/xrt_llamacpp.py
  ensure_port_free refused on any status other than 0, so a probe that timed out (Windows retries a
  refused localhost connect for about 2 s, the probe's timeout) was read as a stale server. No server
  was launched, no prefill ran, no JSONL row was written. Outputs were parked as usual (PARK_OK).
- The queue reported the cells as done; the window is a no-data window, not a measurement.

## Fix (tools only; no measurement logic changed)
- ensure_port_free refuses only on a real HTTP response (status > 0). Status 0 (refused / connection
  error) and -1 (timeout) mean the port is free.
- It now lists running llama-server.exe PIDs (tasklist) and names them in the refusal; if the port is
  free but a llama-server.exe is running, it prints the PIDs to stderr and continues.
- Tests: tests/test_xrt_llamacpp.py (no-HTTP-answer cases free, with and without a PID; HTTP 200/503
  refused with PIDs reported; tasklist parsing).

## Deviation
- The E2-at-74% attempt produced no data. Nothing is discarded; the window stays in the record.
- No prediction, threshold, cell or decision rule changes. E2 at 74% re-runs as registered
  (STEP=e2 SLIDER=74, same cells, same arguments), at the new EXPECT_HEAD.
