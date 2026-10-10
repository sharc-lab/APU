# Paper 1 (ISPASS 2027) one-page outline. Rithwik Sharma, 2026-10-08, updated 2026-10-10

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
3. Silent agent failure and its mechanism: real-run loss **verified** on X2 only (**pending** x2_r2_real_v1b, T2S week
   item 2; Blade real run held by open finding O1). Mechanism **verified** on X2 and Blade, one session per tier
   ([blade-mechanism-4096]); **pending** T2S week item 6, Blade night 2 (e).
4. Predictor **verified** on X2, one session per tier ([R2-shift-predictor-confusion]); mitigation **pending**
   x2_r2_mitigation_v1, T2S week item 3 (Blade night 1 (c) held by O1).

**Top 5 claims**

| claim | evidence | platforms | n / CI | objection, answer |
|---|---|---|---|---|
| Start boundary is one shared heap; first principles predict the flip | "C shared-heap LOO -257 to -26 tokens (-0.11% to -0.01%)" [A-24-flip-point-prediction] | T2S | bounds, no CI | "obvious OOM": device-only accounting is inconsistent [A-24-effective-heap-limit] |
| Device policy sets the default window | llama3.1:8b "t2s_default_ctx": 4096 vs "x2_default_ctx": 131072 [T2S-vs-X2-default-ctx]; Blade "llama3.1:8b: default ctx 4096" [blade-k1-layer-split] | T2S, X2, Blade | deterministic | "known tiering": nobody traced it to agent loss |
| Overflow silent, keeps half | "4096->2050" [ollama-overflow-keeps-half]; Blade "8192 tok -> HTTP 200, evaluated 2050" [blade-k1-layer-split] | T2S, X2, Blade | "all 4 checked cases"; Blade every over-length probe, both models | "llama.cpp feature": shipped default, silent |
| History lost silently; rules lost only at smallest window | "num_ctx_4096 6/6 silent, 6 after window exceeded" [R2-real-v1-gated-kill] | X2 | sessions "of 3" per cell [R2-real-v1-survival] | "one machine": v1b, T2S queued; Blade held by O1 |
| Message drop keeps the system prompt; context shift at the smallest window; token cut discards it (X2) | X2 "34 context shifts ... 1 token cuts" at ollama_ctx_4096 [R2-mechanism-lowlevel]; Blade "system kept on every truncated call True", "token-level cuts 0, context shifts 33" [blade-mechanism-4096] | X2, Blade (token cut X2 only) | one session per tier | "single seed": second vendor now agrees on drop and shift |

Claim count (full table, 18): 11 verified, 4 pending, 3 at risk (was 10 / 4 / 4; the predictor moved from at risk to
verified). Open findings are not counted.

**Open finding O1 (not a claim).** On Blade, llama3.1:8b writes a tool call as text on the final-answer call in the
32768 negative control: "canary misses per session 20260901: 0, 20260902: 0, 20260903: 2, 20260904: 0, 20260905: 2"
[blade-r2-gate-v2], [blade-r2-validation-run2]; FINDINGS 2026-10-09. Not truncation. Blade R2 real and mitigation
held. Pending x2-r2-llama-val-32k (X2 at 32768): no misses = backend/offload effect; text tool calls on the same turn =
window effect.

**Gaps before Nov 1, by impact**

1. R2 on more platforms and seeds: x2_r2_real_v1b; T2S week item 2 (T2S item 1 first); Blade after O1 resolves
   (x2-r2-llama-val-32k).
2. Mechanism: more sessions per tier; T2S week item 6; other Blade tiers, Blade night 2 (e).
3. Mitigation result: x2_r2_mitigation_v1; T2S week item 3.
4. Discrete half: register rows for Blade C1/C2 plus Blade night 2 (d) C3; else drop "three vendors".
5. X2 boundary rows (MX2) plus x2_70b_edge_reps. 6. Predictor scored on held-out observed shifts (analysis only).
