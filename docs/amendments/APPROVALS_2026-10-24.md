# Approvals, Sat Oct 24 2026 (recorded Mon Oct 26 2026, Task BK, verbatim from TASK_BK_MEETING27.md)

## ADDENDUM 1 (Sat Oct 24 2026, ~4:15 PM ET; Zach approved "yes"): ceiling reporting rule
An analysis/reporting rule. It changes no cell, prediction, band or threshold; R-flags are still scored
on the registered L*.
- For every headline ceiling report three numbers side by side:
  - registered L* = the bisection's largest held point (as registered; used for R1-R10 and the law fit);
  - repeat pass rate at L* and L*+step from the matching rep cell (k/n, with run_ids);
  - robust ceiling = the highest point that held in every one of its repeats (or "not established" if
    none).
- Use the other cells' points at the same n as extra pass/fail evidence; say so per number.
- Applied in analysis/cc_20261026/measured.json ("repeats"; pooled over same-arm, same-cap XPS cells at
  the 57% override; a point needs >= 2 attempts).

## ADDENDUM 2 (Sat Oct 24 2026, ~7:55 PM ET; Zach: "thats good if it betters our chances thats fine")
Thesis SCOPE wording fix (accuracy fix allowed under the PLAN_LOCK change rule; no cell, prediction,
band or threshold changes):
- Locked sentence "...which bounds what KV compression can buy to (k+a)/a" becomes scoped to
  whole-context prefill: "the non-KV prefill working set binds whenever a large prefill happens (a cold
  start, a long document or tool prompt, the unchunked runtime default); KV compression is then bounded
  by (k+a)/a, and the prefill granularity (agent turn size or runtime chunk size) decides how much of
  that bound applies." Title unchanged.
- The evidence numbers in ADDENDUM 2 are superseded by ADDENDUM 5 (Mon Oct 26 ~10:20 AM ET; Task BR
  decision 5): last HELD lengths f16 21,736 / 18,532 / 16,429 (e99bb2ad, 71b7d9d8, 9cb921fd), u4 65,206 /
  61,771 / 57,499 (0c7ae0a3, d7e8b43e, 10cb79e4), u4 c256 65,889 (08e52da9), f16 c256 16,473 (2bfae783).

## ADDENDUM 4 and 5 (Mon Oct 26 2026): wording only, no approval quote given in the task file
- Qwen3-4B's trained window is 40,960 tokens: every memory point above it is a "capacity probe", not
  usable context (u4 chunked 64-70K; u4 sessions 57-65K).
- The CC-GROW length correction above.
