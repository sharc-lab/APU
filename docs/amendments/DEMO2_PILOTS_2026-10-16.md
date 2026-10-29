# DEMO-2 pilots P0-P9: decision rules and host commands (registered 2026-10-16)

Status: REGISTERED 2026-10-16 (Task AS), before any pilot runs. Adds to
docs/amendments/DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md; edits nothing.
Runners do not read this file. Its sha256 is in the DEMO2-PILOTS registry
row. Every rule below is fixed before any pilot result is seen.

## 1. Rules for every pilot

- Entries: DEMO-1 pins only (DEMO1_BASE_30, DEMO1_LONGCTX_30, DEMO1_PAIRED_15).
  Never DEMO2_HELDOUT_60 or DEMO2_HELDOUT_60_B (the runner refuses the
  latter without --confirmation-run).
- Model Qwen3-4B INT4 IR, GPU, openvino and genai 2026.2.1, greedy.
- Each pilot is a run of its own with a run_id; no pilot number is a
  DEMO-2 result. Results feed only the arm-specification addendum.
- Ask Zach before every launch. P5 and P7 spend cloud dollars and need his
  approval of the amounts in section 4.
- Analysis: tools/demo2_pilot_analysis.py (P1, P4, P5, P6, P7, P8) applies
  the rules exactly as written here.

## 2. Pilots and decision rules

| pilot | measures | decision rule (fixed now) |
|---|---|---|
| P0 | True-default readback on both hosts: `kv_compiled_readback --mode all` (GenAI's own compile at OPENVINO_LOG_LEVEL=3, IR rt_info, core ports) and a 2-entry `local_only` runner smoke with `-RequireTrueDefault` | Arm A is valid on a host only if all hold: no "Falling back to SDPA" line, compiled KV_CACHE_PRECISION i8, no rt_info KV override, and the runner smoke loads without REFUSED. Otherwise arm A is invalid on that host and DEMO-2 stops there for a decision. |
| P1 | Breakdown of the 130-255 output tokens per local turn into tool-call blocks, think blocks and other text, from raw step text (`-DebugRawText`) | An output-length lever exists if non-tool-call text is >= 50% of local output tokens (character-apportioned estimate). Any lever found is agent-level and is applied to every arm (A-D) or to none. |
| P2 | Decode tok/s at n 3,500 / 12,000 / 20,000 for i8 (default), u4 and f16, both hosts | Bandwidth model holds if measured i8 decode at 12,000 and 20,000 is within +-15% of the model calibrated at 3,500. u4's decode gain is supported if u4/i8 at 12,000 >= 1.05. If either fails, the addendum's latency predictions for B/C are recomputed from P2's numbers, not the model. |
| P3 | XPS resident limit (ladder top 45,000) for chunk cap {4096, 2048} x KV {i8, u4} x cache_size {0, sized}, plus unlimited x sized; cold 10k TTFT with and without the cap; V5-CAL re-run under the latency default | Adopt a chunk cap if its resident limit is >= 20% above the same KV's sealed unlimited limit (i8 20,000 cf3555d5; u4 26,500 e4a22dac) and its cold 10k turn-1 TTFT is <= 10% above the uncapped one. Between 4096 and 2048, the higher limit wins; ties go to 4096. Adopt the sized cache_size if it does not lower the limit. The XPS B/C window = min(40,960, adopted limit - 10%). The DERIVED leap (u4 to 40,960 under a 4096 cap) is supported only if the u4/4096 limit reaches 40,960. |
| P4 | Prompt lookup K in {3, 5} vs off (max_ngram_size 3), u4, same entries | Adopt the K with the larger decode speedup among those with speedup >= 1.1 (median per matched step) and byte-identical raw step text to the off run. Otherwise no prompt lookup. |
| P5 | OBS_MASK K sweep {512, 1024, 2048} on DEMO1_PAIRED_15, arm C settings (u4, window per host) | Choose the lowest $/task among K whose hybrid passes are >= those of K=1024; ties go to the smaller K. |
| P6 | True-default (arm A) `local_only` run: local pass by max n_ctx bucket, n_ctx tail, local outcome classes, step-0 TTFT/decode that would have triggered | Capacity-paradox size: expected passes lost by C vs A = (entries whose max n_ctx is above A's limit and inside B/C's window) x (cloud pass rate 22/30, V2 1acc8c58 - local pass rate in that band). If that exceeds 6 of 60, the addendum sets the XPS B/C window to A's limit - 10% (18,000), so B/C escalate where A does. |
| P7 | A/A: arm A (hybrid_full) twice on DEMO1_BASE_30 + DEMO1_PAIRED_15 (45 entries), per host | Replicates per arm = 2 if discordant entries > 25% of the paired entries (12 of 45; the amendment's ~15 of 60); else 1. d_usd and d_lat set the noise terms of the material-win threshold. |
| P8 | u4 vs i8 (default) `local_only` on DEMO1_BASE_30 + DEMO1_LONGCTX_30: per-turn decoded-call agreement and silent-failure share | Keep u4 in B/C if its silent-failure share among non-escalated entries is <= i8's. Otherwise B/C use the default KV (i8). |
| P9 | Memory part numbers and speed (WMI), active power scheme and AC state, 10-minute sustained decode drift, both hosts | Pin and disclose. If drift (median decode tok/s of the last third vs the first third) exceeds 5% in magnitude, the addendum adds a warm-up rule for every cell. |

## 3. Order

1. P9 and P0 on both hosts (read-only and smoke; $0).
2. XPS local-only ($0): P6 (also the i8 side of P8 and the P1 text), then
   P8 u4, P4, P2, P3.
3. T2S local-only ($0), in its Oct 19-25 window: P2, P4, P8.
4. Spend pilots after approval: P7, then P5, per host.
5. The arm-specification addendum is committed after all pilots and before
   any held-out run.

## 4. Spend (estimates from sealed data; approval needed)

| pilot | host | runs | estimate | cap |
|---|---|---|---|---|
| P7 | XPS | 2 x (BASE_30 + PAIRED_15) | $12.36 | $18.54 |
| P7 | T2S | 2 x (BASE_30 + PAIRED_15) | $6.00 | $9.00 |
| P5 | XPS | 3 K x PAIRED_15 | $13.86 | $20.79 |
| P5 | T2S | 3 K x PAIRED_15 | $4.32 | $6.48 |
| total | | | $36.54 | $54.81 |

Basis (derived/h1_hybrid/demo2/DEMO2_AMEND_TRUE_DEFAULT_PREDICTIONS.json,
"pilots"): BASE_30 at the d482c621 maximum ($1.56, emission_escalate);
PAIRED_15 at the DEMO-1 V3 FULL cost (XPS $4.62, 71d33f7a+ab88f47f; T2S
$1.44, a4284ab0). Caps are 1.5 x estimate per launch. The gold-trajectory
model predicts much more for these jump-selected entries ($29/run on the
XPS) and was not used for caps; --max-usd stops a run at its cap.

## 5. Host commands (in order; each needs Zach's OK)

Notation. XPS = `ssh <user>@<host>`, repo `C:\Users\<user>\Projects\gnn-hls-accel`.
T2S = `ssh <user>@<host>`, repo `C:\Users\<user>\Projects\gnn-hls-accel`.
`<R>` = the repo path of that host. `<PY>` = `<R>\.venv-seam\Scripts\python.exe`.
`<C>` = `C:\ovcache\qwen3-4b` (CACHE_DIR, created by the first load).
Every launch_h1 command is run first with `-DryRun`, then without it.
$0 = local only, no cloud.

P9 ($0, both hosts; read-only):
1. `XPS 'Get-CimInstance Win32_PhysicalMemory | Select-Object Manufacturer,PartNumber,Speed,ConfiguredClockSpeed,Capacity | Format-List; powercfg /getactivescheme; Get-CimInstance Win32_Battery | Select-Object BatteryStatus,EstimatedChargeRemaining'`
2. The same on the T2S.
3. Drift ($0), per host: `XPS 'cd <R>; git pull --ff-only; powershell -NoProfile -File tools\spawn_detached.ps1 -CommandLine "<PY> -u tools\run_decode_match.py --arm gpu_only --model-spec configs\models\Qwen3-4B-int4-ov.yaml --n 3500,3500,3500,3500,3500,3500,3500,3500,3500,3500,3500,3500" -LogPath derived\c2_ttft\_launches\demo2_p9\drift.log -WorkingDirectory <R>'` (12 points x 3 repeats, about 10 minutes; DERIVED from 27 tok/s).

P0 ($0, both hosts):
4. `XPS 'cd <R>; git pull --ff-only; powershell -NoProfile -File tools\kv_compiled_readback.ps1 -Mode all'`
5. `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Interleaved -InterleavedPolicies local_only -PolicyCapUsd "local_only=0" -SessionMaxUsd 1 -Placement gpu_only -Kv default -Residency RESIDENT -RequireTrueDefault -CacheDir <C> -Entries derived\h1_hybrid\demo1\DEMO1_BASE_30.json -EntryCount 2'`
6. Steps 4-5 on the T2S (add `-T2S` to launch_h1; watchdog idle, Rithwik OK).

P6 + P1 + P8 i8 side ($0, XPS; arm A local only, raw text and memory on):
7. `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Interleaved -InterleavedPolicies local_only -PolicyCapUsd "local_only=0" -SessionMaxUsd 1 -Placement gpu_only -Kv default -Residency RESIDENT -RequireTrueDefault -CacheDir <C> -DebugRawText -MemTelemetry -GpuMemCounters -Entries derived\h1_hybrid\demo1\DEMO1_BASE_30.json'`
8. Step 7 with `-Entries derived\h1_hybrid\demo1\DEMO1_LONGCTX_30.json`.
9. Analysis: `XPS 'cd <R>; <PY> tools\demo2_pilot_analysis.py p6 <run dir of 8>'` and `... p1 <run dir of 7> <run dir of 8>`.

P8 u4 side ($0, XPS):
10. Steps 7-8 with `-Kv u4 -CompiledReadback` instead of `-Kv default -RequireTrueDefault`.
11. `... demo2_pilot_analysis.py p8 <run dir of 7> <run dir of 10 (BASE_30)>` and the same for LONGCTX_30.

P4 ($0, XPS, then T2S):
12. Off: `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Interleaved -InterleavedPolicies local_only -PolicyCapUsd "local_only=0" -SessionMaxUsd 1 -Placement gpu_only -Kv u4 -Residency RESIDENT -CompiledReadback -CacheDir <C> -DebugRawText -Entries derived\h1_hybrid\demo1\DEMO1_BASE_30.json -EntryCount 10'`
13. On, K=3: step 12 plus `-PromptLookup -NumAssistantTokens 3 -MaxNgramSize 3`.
14. On, K=5: step 12 plus `-PromptLookup -NumAssistantTokens 5 -MaxNgramSize 3`.
15. `... demo2_pilot_analysis.py p4 <dir 12> <dir 13>` and `p4 <dir 12> <dir 14>`.

P2 ($0, XPS, then T2S; one detached run per KV, gates and canary inside the tool):
16. `XPS 'cd <R>; powershell -NoProfile -File tools\spawn_detached.ps1 -CommandLine "<PY> -u tools\run_decode_match.py --arm gpu_only --model-spec configs\models\Qwen3-4B-int4-ov.yaml --n 3500,12000,20000" -LogPath derived\c2_ttft\_launches\demo2_p2\i8.log -WorkingDirectory <R>'`
17. Step 16 with `--arm gpu_only_u4` (log u4.log) and with `--arm gpu_only_f16` (log f16.log; f16 at 20,000 is above its 15,000 resident limit and may fail, which is recorded).

P3 ($0, XPS only; each cell is one detached run of about 50 minutes, DERIVED: 9 rungs x the 46k cold turn 240 s of 2b3316b6 plus canary 752 s; two cells per boot window):
18. `XPS 'cd <R>; powershell -NoProfile -File tools\spawn_detached.ps1 -CommandLine "<PY> -u tools\run_resident_limit.py --arm gpu_only_u4 --model-spec configs\models\Qwen3-4B-int4-ov.yaml --high 45000 --sched-max-batched-tokens 4096" -LogPath derived\c2_ttft\_launches\demo2_p3\u4_c4096.log -WorkingDirectory <R>'`
19. Step 18 with `--sched-cache-size 2` added (log u4_c4096_s2.log).
20. Step 18 with `--arm gpu_only` (i8) (log i8_c4096.log), and with `--arm gpu_only --sched-cache-size 4` (log i8_c4096_s4.log).
21. Step 18 with `--sched-max-batched-tokens 2048` for `gpu_only_u4` and `gpu_only` (logs u4_c2048.log, i8_c2048.log).
22. Unlimited x sized: step 18 with `--sched-max-batched-tokens 18446744073709551615 --sched-cache-size 2` (u4) and `--arm gpu_only ... --sched-cache-size 4` (i8).
23. Cold 10k cost: `XPS 'cd <R>; powershell -NoProfile -File tools\spawn_detached.ps1 -CommandLine "<PY> -u tools\run_warm_kv.py --arm gpu_only_u4 --model-spec configs\models\Qwen3-4B-int4-ov.yaml --n-cached 10000 --delta-tokens 183" -LogPath derived\c2_ttft\_launches\demo2_p3\cold10k_u4_default.log -WorkingDirectory <R>'`, then the same with `--sched-max-batched-tokens 4096` (and 2048 if adopted candidates need it). Turn-1 TTFT is the cold 10k figure.
24. V5-CAL re-run under the latency default (registry V5-CAL-2; this file is its prereg note; the code already builds a plain LLMPipeline): `XPS 'cd <R>; powershell -NoProfile -File tools\launch_v5_cal.ps1 -Detach -Rehearsal'`, then without `-Rehearsal`.

P7 (cloud spend, after approval; per host, run twice):
25. `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Policy hybrid_full -Placement gpu_only -Kv default -Residency RESIDENT -RequireTrueDefault -CacheDir <C> -MemTelemetry -Entries derived\h1_hybrid\demo1\DEMO1_BASE_30.json -MaxUsd 2.34'`
26. `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Policy hybrid_full -Placement gpu_only -Kv default -Residency RESIDENT -RequireTrueDefault -CacheDir <C> -MemTelemetry -Entries derived\h1_hybrid\demo1\DEMO1_PAIRED_15.json -MaxUsd 6.93'`
27. Steps 25-26 again (replicate 2). On the T2S: the same with `-T2S` and `-MaxUsd 2.34` / `-MaxUsd 2.16`.
28. `... demo2_pilot_analysis.py p7 <replicate 1 dir> <replicate 2 dir>` per entries file; pool the two files' counts.

P5 (cloud spend, after approval; per host; XPS window 23,850 provisional, T2S 40,960):
29. `XPS 'cd <R>; powershell -NoProfile -File tools\launch_h1.ps1 -Policy hybrid_full -Placement gpu_only -Kv u4 -Residency RESIDENT -CompiledReadback -CacheDir <C> -ObsMaskK 512 -ContextWindow 23850 -Entries derived\h1_hybrid\demo1\DEMO1_PAIRED_15.json -MaxUsd 6.93'`
30. Step 29 with `-ObsMaskK 1024`, then `-ObsMaskK 2048`. On the T2S: `-T2S`, `-ContextWindow 40960`, `-MaxUsd 2.16` each.
31. `... demo2_pilot_analysis.py p5 <K512 dir> <K1024 dir> <K2048 dir>`.
