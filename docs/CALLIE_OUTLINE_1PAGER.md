# Paper 1 (ISPASS 2027) one-page outline. Rithwik Sharma, 2026-10-08, updated 2026-10-14

Full outline: `docs/PAPER1_DRAFT.md`. Numbers carry register ids from `docs/NUMBERS_REGISTER.md`. Deadline and
page limit unconfirmed (Dec 2 vs mid-Dec estimate; 10 pages assumed).

**Title:** *Silent by Default: How Memory Architecture and Runtime Policy Shape Local LLM Agents on AI PCs*

**Abstract (150 words):** Local agent stacks on consumer AI PCs fail in ways their users cannot see. We characterize
three platforms from three GPU vendors spanning unified and discrete memory: an Intel integrated GPU, an AMD Strix
Halo integrated GPU, and an NVIDIA laptop GPU. First, memory architecture sets hard boundaries: on the Intel
integrated GPU a model start succeeds or fails at one shared-heap limit that also counts host-visible buffers, and a
first-principles accounting predicts the measured flip point; on discrete memory, overflow becomes a silent spill.
Second, runtime policy, not hardware, sets the effective context window: one runtime picks very different default
windows on two integrated GPUs because of device detection, and truncates overflow silently. Third, these defaults
reach agents: in tool-using sessions, history is lost without any error, and at the smallest window the system
prompt is discarded. We give a predictor for that loss and evaluate a pre-registered mitigation.

**Contributions**

1. Cross-vendor memory boundaries: T2S shared-heap flip point **verified**; X2 spill/crash **pending** x2_70b_edge_reps;
   Blade spill **pending** Blade night 2 (d).
2. Runtime policy sets the effective window, overflow silent: **verified** T2S + X2 + Blade (Blade night 1 (a), K1,
   [blade-k1-layer-split]).
3. Silent agent failure and its mechanism: real-run loss **verified** on X2 only, now 3 models x 5 seeds x 5 tiers
   (v1 + v1b; **pending** T2S week item 2; Blade real run held by O1). Mechanism **verified** on X2 and Blade, one
   session per tier ([blade-mechanism-4096]); **pending** T2S week item 6, Blade night 2 (e).
4. Predictor **verified** on X2, one session per tier ([R2-shift-predictor-confusion]). Mitigation **run** on X2:
   the obvious fix works for one model and fails for another at the smallest window, pre-registered verdict refuted
   ([R2-mitigation-v1-verdict]); T2S week item 3 pending (Blade night 1 (c) held by O1).

**Top 6 claims**

| claim | evidence | platforms | n / CI | objection, answer |
|---|---|---|---|---|
| Start boundary is one shared heap; first principles predict the flip | "C shared-heap LOO -257 to -26 tokens (-0.11% to -0.01%)" [A-24-flip-point-prediction] | T2S | bounds, no CI | "obvious OOM": device-only accounting is inconsistent [A-24-effective-heap-limit] |
| Device policy sets the default window | llama3.1:8b "t2s_default_ctx": 4096 vs "x2_default_ctx": 131072 [T2S-vs-X2-default-ctx]; Blade "llama3.1:8b: default ctx 4096" [blade-k1-layer-split] | T2S, X2, Blade | deterministic | "known tiering": nobody traced it to agent loss |
| Overflow silent, keeps half | "4096->2050" [ollama-overflow-keeps-half]; Blade "8192 tok -> HTTP 200, evaluated 2050" [blade-k1-layer-split] | T2S, X2, Blade | "all 4 checked cases"; Blade every over-length probe, both models | "llama.cpp feature": shipped default, silent |
| History lost silently; after-window silent failures rise as the window shrinks; default tier has none after the window | "num_ctx_4096: overall 10/15 vs default 0/15, excess +0.667 (95% CI +0.533 to +0.800)", "num_ctx_8192: overall 8/15 vs default 0/15, excess +0.533 (95% CI +0.333 to +0.667)" [R2-v1b-excess-over-default]; default "gated 3/15 silent, 3 baseline-type" [R2-v1b-default-baseline-type] | X2 | 3 models x 5 seeds per tier; bootstrap over sessions, 10000 resamples [R2-v1b-excess-over-default] | "default fails too": all default failures are baseline-type, 0 after the window; T2S queued; Blade held by O1 |
| The obvious fix works for one model, fails for another at the smallest window | "num_ctx_4096 llama3.1:8b recovered; num_ctx_4096 qwen3:14b not_recovered (rules below 90%: rule2_log_event_called)" [R2-mitigation-v1-verdict] | X2 | 2 models x 3 seeds, no CI | "you tuned until it worked": pre-registered, and refuted |
| Message drop keeps the system prompt; context shift at the smallest window; token cut discards it (X2) | X2 "34 context shifts ... 1 token cuts" at ollama_ctx_4096 [R2-mechanism-lowlevel]; Blade "system kept on every truncated call True", "token-level cuts 0, context shifts 33" [blade-mechanism-4096] | X2, Blade (token cut X2 only) | one session per tier | "single seed": second vendor now agrees on drop and shift |

Claim count (full table, 19): 13 verified, 3 pending, 3 at risk (was 11 / 4 / 3 of 18; the mitigation moved from
pending to verified as a refuted pre-registered hypothesis, and the O1 confound result is new claim P1-19). Open
findings are not counted.

**Open finding O1, partly resolved.** On Blade, llama3.1:8b writes a tool call as text on the final-answer call in the
32768 negative control: "canary misses per session 20260901: 0, 20260902: 0, 20260903: 2, 20260904: 0, 20260905: 2"
[blade-r2-gate-v2], [blade-r2-validation-run2]. Confound check on X2 at 32768, same model, seeds and settings: "canary
misses 0 of 20 canaries over 5 sessions" and "verdict: backend or offload effect" [x2-r2-llama-val-32k]. So per the
pre-registered rule the Blade misses are a backend (CUDA) or CPU-offload effect, not the window (claim P1-19). Still
open: backend vs offload (Blade at 32768 is partly on the CPU, [blade-k1-layer-split]); the follow-up that separates
them is pending, not yet designed or queued. Blade R2 real and mitigation stay held until then.

**Gaps before Nov 1, by impact**

1. R2 on more platforms: T2S week item 2 (T2S item 1 first); Blade after the O1 backend-vs-offload follow-up; H-scale
   (30B to 70B) on X2.
2. Mechanism: more sessions per tier; T2S week item 6; other Blade tiers, Blade night 2 (e).
3. Mitigation: more models and seeds; T2S week item 3.
4. Discrete half: register rows for Blade C1/C2 plus Blade night 2 (d) C3; else drop "three vendors".
5. X2 boundary rows (MX2) plus x2_70b_edge_reps. 6. Predictor scored on held-out observed shifts (analysis only).
