# Approvals, Sun Oct 25 - Mon Oct 26 2026 (Task BR)

Recorded Mon Oct 26 2026, about 10:05 AM ET (Task BR), verbatim from
~/seam_closeout_1013/papers_20261015/TASK_BR_WEEK1_AMENDMENTS.md (Part 0) and Zach's Task BR message.

## RED FLAG R2 (Sun Oct 25 2026), as stated in the task file
- R2 fired (Sun Oct 25): balloons null (598e882d, f9953743, 0014465c, 62d59ab8) and A25 slider inverse:
  u4 22,750 / 21,250 / 17,750 at 50 / 57 / 74% (a2a7c763, f4abaf6e, 9e31d4cb); f16 13,250 / 12,500 / failed at
  8,000 (769c5a03, c461f165, 39d38c85 unsealed: low rung failed). Override max on this XPS = 74% (range 26-74%,
  Zach's screenshot; setting is "Shared GPU/NPU Memory Override"); restored to 57% after the sweep.
  Session drift: u4 at 57% = 25,000 (Fri) vs 21,250 (Sun).

## Decision (verbatim)
Zach, Sun Oct 25 2026 ~10:50 PM ET: "ok on all" to:
1. amendments INV, XRT, SESS, STAT, 8B as in the report;
2. a pinned llama.cpp Windows build (SYCL and Vulkan release zips) installed in our own folder on the
   XPS, plus one pinned Qwen3-4B GGUF (official Qwen repo; Q4_K_M or the closest to our INT4, record
   sha256) - these two downloads are approved, nothing else;
3. asking Rithwik for RTX 4070, Strix Halo and 64 GB-box time;
4. registering the Qwen3-8B a prediction before the M2 cells run;
5. the report's direction (lead with the KV-precision x chunking interaction and the one-budget
   result; gate the inversion at Gate A Wed Nov 4; drop DEMO-2/concurrency/bisection maxima as
   headlines). Title stays as locked until the Nov 15 framing checkpoint; the report's candidate
   ("Beyond the KV Cache: Prefill Granularity Sets the Context Ceiling of LLM Agents on AI PCs") is
   recorded as the leading proposal.

## HYB-1 (Mon Oct 26 2026, about 9:41 AM ET, Zach's Task BR message, verbatim)
"Run ~/seam_closeout_1013/papers_20261015/TASK_BR_WEEK1_AMENDMENTS.md end to end, including Part 4
(HYB-1). Read every file it lists first, including
~/seam_closeout_1013/research_20261021/GPU_share_inversion_and_acceptance_plan.md and
Hybrid_execution_arm_impact_plan.md. ... Record the approvals verbatim, register
INV/XRT/SESS/STAT/8B/HYB-1 with predictions fixed before any run, write the new queue scripts ...
launch nothing, do not open any trace data."
- Note: the task file on disk (saved 9:40 AM ET) has Parts 0-3 and no Part 4. HYB-1 is registered
  from the "What HYB-1 must freeze" section of Hybrid_execution_arm_impact_plan.md. The plan proposes
  filing by Fri Oct 30 with owner sign-off; it is filed now on this instruction, before any trace row
  is read. Its thresholds are the plan's, as written.

## Scope
- Not approved by this: any other download or install, any spend, any T2S change, any launch by
  Claude Code. Zach launches every queue. Nothing already registered is rescored or changed.
