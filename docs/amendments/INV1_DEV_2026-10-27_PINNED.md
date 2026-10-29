# INV-1 deviation: pinned balloons not run in boots 2-4 (Tue Oct 27 2026)

Recorded Tue Oct 27 2026, about 10:15 AM ET, by Zach's instruction, before INV-1 boot 2 starts.
Registration: docs/registrations/BR_INV_2026-10-26.md (unchanged). Predictions:
derived/br/BR_INV1_PREDICTIONS.json (unchanged).

## What happened in boot 1 (74%, window_20261026_224645_inv1_b1_p74)
- Pinned GPU 1 GiB balloon, cell bal_pin_gpu1_u4, run_id cf47fcf7-8e13-4b28-b745-93f471706a7c (rc=1):
  incompatible with the quiescence gate by design. The balloon keeps its buffer resident with a kernel
  every 10 ms, so the GPU is never quiet when the gate checks it.
- Pinned 2 GiB CPU balloon (VirtualLock), cell bal_pin_cpu2_u4, run_id
  5a727658-8ed3-43b3-b35c-f05c316d99ab (rc=1): hung about 8 h (queue alert at 04:27 AM ET: still
  running after 7,200 s), then refused AC_offline.
- Neither run yields a P-INV-4 measurement. Both stay in the record as run; nothing is discarded.

## Deviation
- P-INV-4 (H3) is NOT RUN in boots 2-4 and is deferred. Steps 31-32 of
  ~/seam_closeout_1013/xps/run_queue_xps_inv1.sh are skipped with SKIP_PINNED=1 (new option, default 0;
  the default behaviour is unchanged).
- No other prediction, threshold, cell or decision rule changes. P-INV-1..3 and 5..7 run as registered.
- Consequence, not a rule change: until P-INV-4 runs, the "H3 if P-INV-4 holds" branch of the
  mechanism-class rule cannot be assigned at Gate A (Wed Nov 4).
- A pinned-balloon design that can coexist with the quiescence gate, if any, needs its own registration
  before it runs.
