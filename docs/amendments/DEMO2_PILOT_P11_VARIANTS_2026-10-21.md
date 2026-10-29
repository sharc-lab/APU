# P11-HANG-REGIME addendum note: variant inputs pinned, run options, T2S order (Task BA)

Written 2026-10-21 (Task BA), BEFORE any P11 run. It adds to
docs/amendments/DEMO2_PILOT_P11_HANG_REGIME_2026-10-20.md and edits nothing in it.
- No decision rule, classification or threshold changes.
- It records the variant inputs that registration required to be pinned
  before any phase-M or phase-S cell, and the run options.
- Approved by Zach 2026-10-21 (Task BA decision 2): run P11 on the T2S
  before the hand-back, REBOOT_OK=1. Tailscale on the T2S runs in
  server/unattended mode under EVO-T2S\SHARC (confirmed); py-spy is approved
  on the T2S too, fetched the same standalone way (tools/get_pyspy.ps1,
  pip freeze hash checked).

## Inputs
- Base: derived/repro_hang123/inputs_e123_t0.json, sha256
  d12142f89702f0ab92a48148abaa034760bb20c87558b58945a92a29a405f2f0 (L1 and M6).
- Variants: derived/repro_hang123/variants/ (builder
  tools/repro_hang123_variants.py, `tools/repro_hang123.py variants`), index
  INDEX.json sha256 . HF tokenizer pinned as in Task AY.

## How the registered definitions were built (interpretations, fixed now)
- S is known only after phase L, so the full set is built for both possible
  stalling steps. Turn 0 of 123 has two gold steps, so the sets are S1 and
  S0. The window script uses the set of the S that phase L finds.
- S = 1 (expected): the replaced body is the largest tool response at step 1.
  That is the get_stock_info result, 31,705 chars, two lists of 2,000 floats.
  N_body = 31,665 HF tokens, measured on the body alone (registration's N, by
  prompt arithmetic, is about 31,700). Prefix and suffix are kept byte for
  byte.
- S = 0: N = prompt tokens. The body is everything between
  "<|im_start|>system\n" and the final "<|im_end|>\n<|im_start|>assistant"
  (2,177 tokens). There is no tool response, so M5 (random floats in 123's
  format) is not defined for S0 and is not built.
- M1-M3 sources (rule: largest single gold-path tool result, file order,
  123 excluded):
  - multi_turn_long_context_125 (31,666 tokens), _141 (31,666), _100
    (31,665);
  - all three are get_stock_info results of the same generator, so M1-M3
    are real content of the same kind and size as 123's.
- Real sweep at S1 rescales 123's own lists: prefixes for f < 1; extended by
  repeating their own values for f > 1. JSON stays valid. Real sweep at S0
  is a token-boundary prefix of the body (f < 1) or the body repeated (f > 1).
- Random sweep: the M4 token stream (seed 20261020) at f x N_body.
- All bodies are within +-0.5% of the target.
- Model limit: S1 1.25x and 1.5x (real and random) put the prompt plus 512
  new tokens above Qwen3-4B's 40,960 positions. They run as registered. An
  error before 600 s is recorded as "error (above model max)", not as a hang
  and not as a completion. They do not count toward any class.

## Repro run options (harness, Task BA; default off)
`tools/repro_hang123.py run ... --watchdog-s 600 --out-dir
derived\repro_hang123\runs --label <cell> [--py-spy <exe>]`:
- the runner's TurnWatchdog is armed per generate;
- on expiry: Python stacks, py-spy --native when given, hang_events.json, and
  exit 86 (the process ends itself; nothing is killed from outside);
- run.json records every step (pre-generate time, done, wall, output sha256).
- M/S cells use `--only-step S` (cold prefill of step S alone). L1 runs all
  steps in order.

## T2S order (run_window_t2s_hang.sh), non-hanging first where known
1. L1 repro (all steps), L2 runner (arm A, entry 123 alone, -StepLog).
2. P10-T2S C2, C3 (DEMO2_PILOT_P10_T2S_2026-10-21.md).
3. M4, M1, M2, M3, M5, then S in ascending f (random before real at each f),
   then M6 last, as registered.
4. P10-T2S C4, C5, C6, C1.

If neither L1 nor L2 hangs, M and S are not run (class "not reproduced"). If
only L2 hangs, S is the step of L2's last step_log line, and M/S use the
repro (a recorded limit). A reboot follows every hang. The script stops at
10:00 PM ET 2026-10-25 whatever its state.

## Variant sha256 (every input)

### Set S1 (stall at step 1; real body 31,665 HF tokens, 31,705 chars)

| cell | file sha256 | prompt sha256 | body tokens (target) | prompt tokens | > 40,960 with 512 new |
|---|---|---|---|---|---|
| M1 | d300dca8dec79a6f3d7c40cbe82c81f755e54947e98da1ec0dcd7ed710fbaec8 | 86f9849da30e6a667eff6d90c73e80a2c56e023ad9f31c2b132cb252e112ff44 | 31,665 (31,665) | 34,051 | no |
| M2 | 0efcb216316b56c55532277e2c85567557ad76fdf75d327b1820e7911494ef61 | 62760185acde333830fc51672a2c16ce2d8451f142a231d6ce518f9f810c611e | 31,665 (31,665) | 34,051 | no |
| M3 | 570dc33ae4e9ec4521a0e797cc4ca4f7a9166832bd064c887323347cb4279425 | 4ca2639f9d439e72e2bc1200701f04e041b0362326645b6552dd7f01d54ec285 | 31,665 (31,665) | 34,050 | no |
| M4 | 0036beead726eb1db0780ad59353c4f25292ee77381513d0c7de1b3cc9c34e2e | 33969b2853304176b56e2dfe3b871475608184a2e28d19ea9dfb87c071a6a0b0 | 31,714 (31,665) | 34,100 | no |
| M5 | fe3dc4747a407e9e7b05dd82511a80e8fb70c317dc93f1916e972b8e55ffadd6 | c349bb99a2383a1abb78f038b605a7a9cd4311f9958371edbd04b5a4a725f397 | 31,822 (31,665) | 34,207 | no |
| S_rand_0p50 | 781911d3106ccca37c6bea6840ab67a5420b81dbc704a30562ce2eca0ec73200 | adb4f099e93f89472c162e4f2b0756a54bc9ff84e443dee10bc0c2078a3362ba | 15,826 (15,832) | 18,212 | no |
| S_rand_0p75 | ff29cb6e78221870a376de7ff5ebdd3485d223b81b8d5bc6782b2f0da936939d | 7e91c671167c47bd2d5608c90ec303e57c0941504423daa31d5ec2cbbbe289c9 | 23,860 (23,749) | 26,246 | no |
| S_rand_0p90 | 473917f43cc5c39830ddc5a987b2fe0feff039bde40148c5852c1c0e7c835eaa | c46442efdebf3d91f487591cb17b73baf1755623f5e499560b2d6ffd64dfeb8f | 28,382 (28,498) | 30,768 | no |
| S_rand_1p00 | b69752021ec660206864f3dfddc6b1101715016681c7acbc4601c85c90479a4e | 33969b2853304176b56e2dfe3b871475608184a2e28d19ea9dfb87c071a6a0b0 | 31,714 (31,665) | 34,100 | no |
| S_rand_1p10 | e721025dc7a35114dec35acf21278ae2334a2db355fdae8eb66c784eb2ef5408 | 50f9e4084b06d30bdacd53e35f32e047e5e14edc54e3e97cff566a703c9c8f0d | 34,977 (34,832) | 37,363 | no |
| S_rand_1p25 | dc95654af93b070749abf6640ab4c7a331608174d24fff8f76b68de2944e07a3 | b2bc3fea98434c2073ac55f29d9473df8a4132a68e4731ad2c0e211232f62e70 | 39,513 (39,581) | 41,899 | yes |
| S_rand_1p50 | aa7c131ad51dc9c1f294ea66dbb05163830acbcf7946092cda544cb662ac3e06 | 6f92e8bcc792080711ea5c344fd04bffead55615b637836ab3ee7af769ee86ea | 47,360 (47,498) | 49,746 | yes |
| S_real_0p50 | 5915760c6a1c592a745806cb7800e23f444516413762d1a680967a7e85dd0846 | 21b73995b1f8bb89b1055180f67fe2860ce99f3fdf7c55f95e02069e3480844c | 15,910 (15,832) | 18,295 | no |
| S_real_0p75 | 81e862fae14cd92b55f5196d56c91dafebccc3d984327a77dc8e5e9e8a39ce9e | 12779d6b312bd0149569b7bf93baabcf9e816ed2088410c2209a93fb9b075f99 | 23,858 (23,749) | 26,243 | no |
| S_real_0p90 | a183f5c2442fc8c5dd8df1433b01e49d1b846d00930cb581d9f0e68194d7d00d | be88d5e5a7f96c4b23922eb1252c98275374c5bf134d98c703306d5a18f30ff0 | 28,622 (28,498) | 31,007 | no |
| S_real_1p00 | 88bcbc5d4c508dfde51b5c7f680697af3d76da86a64268c9cd7b82c147328400 | 4e965f37fee49f8eb7ad764c7330e371f933e73923737c3724d4cab9c2e022c7 | 31,665 (31,665) | 34,050 | no |
| S_real_1p10 | 412e9d55371049ec7b00c581760f0c75ccdc93c42a3b3a8cb0b9cdf3839bccc9 | 5197adef517068f5cd0f4eaf25772309047ad84cdf2f85574a8b7a779ca5002b | 34,973 (34,832) | 37,358 | no |
| S_real_1p25 | 2397919d0f357424e733357c3ebe400176bb05c03f210800db8905657fc64e8e | c82d8cfa4f8869f6d1a5e47999f00a57c3ec5bf8818471b1484c84e21d6a05f8 | 39,722 (39,581) | 42,107 | yes |
| S_real_1p50 | 5b68148bd2f687f008399dcc8b602c02ed3fd117023bafb71df3e0ff934b8750 | 7d11d18e89957215c4bd7d74ed5a77d7276db74c923acbba52848f0999fde431 | 47,660 (47,498) | 50,045 | yes |

### Set S0 (stall at step 0; real body 2,177 HF tokens, 9,293 chars)

| cell | file sha256 | prompt sha256 | body tokens (target) | prompt tokens | > 40,960 with 512 new |
|---|---|---|---|---|---|
| M1 | 22593331bbd7444217fd0eab2f36893adb19609cd23651420f70171d751b1b61 | 7391b447a8312b3816c3276d8ea558883f4b4db95baf8734f9e0667ed0098106 | 2,177 (2,177) | 2,189 | no |
| M2 | 9f5d731216bfe7e01e1fc349126f8a278f5acaa30a55f6c02b6de7fa3700429f | 127ba383bdfd44070c31b1f519ecf3a0ac81aa259c97c56bf42d32a3e8a30e18 | 2,177 (2,177) | 2,189 | no |
| M3 | 43eac4a90069b1b4d0f54652d77f461fc782cfca35a92b5a24b77dc62317f2f7 | 7cbf95bf2f1e17b53742ef997a5c9e3317e192398b13afe7a344add689831c2f | 2,177 (2,177) | 2,189 | no |
| M4 | 66267ebff24d6ce716f5e9bb3cdca1dcdb0230c7c753587da8f2ce6194ce3b69 | 325f76cd449e997e0d3437e3c13805b2e0cf53f2ce4aab8ed12f2b1cdd350d64 | 2,185 (2,177) | 2,197 | no |
| S_rand_0p50 | 3a8a8190b69e8f9c43c5f4e6dd7d7de263033aa3237bc31680240b552f9f0068 | 6c5c213c25de4daf08e63da208d6a73a67618527db57c3e62bf2507918e107ad | 1,093 (1,088) | 1,105 | no |
| S_rand_0p75 | 58eb9c32c701e159f6a3be8b05dacbf5a032eacef3355ee985661ebecf068d6a | 20afaea38d5e2591cf08322a15e5e04bb9da9a7b021eb93a17886bd468bcc8e8 | 1,625 (1,633) | 1,637 | no |
| S_rand_0p90 | f8c556920612f9d6b486a5dd9f7da645c3b3e6297ca192deb650560ad99106c7 | 1101cb676ddecabf938c7db25ee32c527c598ba35c2f4fad7898779b538dede3 | 1,959 (1,959) | 1,971 | no |
| S_rand_1p00 | 44b2cbe66570af3fca6512172b75078156b99de23f33646c90a56eab36c5d3c8 | 325f76cd449e997e0d3437e3c13805b2e0cf53f2ce4aab8ed12f2b1cdd350d64 | 2,185 (2,177) | 2,197 | no |
| S_rand_1p10 | c5195fba150d11c636bea8bf7c46839008cb9f5050ea816c5c14949de0cadbd9 | 7c743b6ef06e0b5ad7a8d8514767457762ce6a37abc470a17f46b8ffa3cdcf6d | 2,395 (2,395) | 2,407 | no |
| S_rand_1p25 | f3cb369b0a7590cd19ba779997402741bd6f12ac8bc9ad57253e2f81e0607e1d | 20285c26c71d050dac286ba4e618e9c07f0d0ab3d9d17d8701d3c7e7b8114993 | 2,721 (2,721) | 2,733 | no |
| S_rand_1p50 | a88833039b6a6d5b2a810df0ee20f3e3d576d292a5197cc67105810dc376bbea | 84777072db77c05e993516b0395d1d341b25bb4c8d1a389e19cc384b96a08bea | 3,268 (3,266) | 3,280 | no |
| S_real_0p50 | 618f91f80088916408fddec72a6732d41fc4684f286e93f4e354133a7a993b83 | c06d2c21f4d9a138cb79ae5c898e086b4ca75a4161fced4421fa4d62550d0bfa | 1,088 (1,088) | 1,100 | no |
| S_real_0p75 | 87a7561da4eb236165b5417c43ce0ee70f07eaaffeb3b6b8b3d931eaa5a79aae | 177969bc96e15c109a6bb13264b3dbeecc8cf368b9a7f3512c2a4ea0e66cac6c | 1,633 (1,633) | 1,645 | no |
| S_real_0p90 | fe1779a1ee9d937ef2cda1f7d5010ec0a4d599a18868986ce50098327316412c | 261baaac79e976549e7175f5f65f23345de7b3bd1c0007cfe775a1cbffc8971a | 1,959 (1,959) | 1,971 | no |
| S_real_1p00 | d108e0e66d0d6e6db9a1325778a20b3c6c87ca859a765918ffd1cce9e72ce24e | d3b347bc19c601c33c964762d5579cfc9a5ebb2020d286ecea583d11bcaa96a9 | 2,177 (2,177) | 2,189 | no |
| S_real_1p10 | 80cee2f702f8464467600097d9cde7c3b60d75b57d28ad6b18e5a341c890bbb8 | ef6b6ad51224a7597a4c2e3220233e2e9978ffcc7b1b9880bef50964543ea17d | 2,395 (2,395) | 2,407 | no |
| S_real_1p25 | a2c8518f7ae67262f57b336d2103eeab5e7e7eb3de524734a39684e845ad47ee | 52b41d346892e79c9b0f7172bddc130e097ac83e892a7afd949fee436bee1191 | 2,721 (2,721) | 2,733 | no |
| S_real_1p50 | 6df2ac45efcbc834b3043d21e84bd000ee77ced59fe81b035155bd8f364cf585 | 8b39bf0693783cde50db06b979d3e8afabaf3575b851d1617f5a767cc80e8128 | 3,266 (3,266) | 3,278 | no |
