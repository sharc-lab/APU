# Model pin registry

v1 pins under this directory may record an **incomplete** `publisher_quantization`
(missing `awq` / `scale_estimation` / `dataset`, or lacking explicit nulls). Those
v1 YAML files are **frozen**: sealed `raw/*/manifest.json` entries store
`model.provenance.spec_sha256 = sha256(pin_file)`, so editing a v1 pin would
disagree with every historical seal that recorded it.

**v2 is authoritative for recipe.** Same IR bytes (`ir_sha256` / `ir_files` /
`revision` / `verification` copied from v1 — not recomputed). Prefer v2 (or a
complete first-class pin such as int8) when selecting a weight IR for new work.

| file | ir_sha256 | revision | ladder_position | recipe | supersedes | superseded_by |
|------|-----------|----------|-----------------|--------|------------|---------------|
| `Qwen3-4B-int4-ov.yaml` | `c1821f29332faa48871c7f16426a21443fbc701fa4e4c8a581a8c51ab7bf2cb2` | `b467368d16b75df14055562fe927ae8e1f15f7ef` | slice-4b | INT4_SYM ratio=1.0 group_size=128; **incomplete** (no awq / scale_estimation / dataset) | — | `Qwen3-4B-int4-ov.v2.yaml` |
| `Qwen3-4B-int4-ov.v2.yaml` | `c1821f29332faa48871c7f16426a21443fbc701fa4e4c8a581a8c51ab7bf2cb2` | `b467368d16b75df14055562fe927ae8e1f15f7ef` | slice-4b | INT4_SYM ratio=1.0 group_size=128 awq=true scale_estimation=true dataset=wikitext2 | `Qwen3-4B-int4-ov.yaml` | — |
| `Qwen3-8B-int4-ov.yaml` | `cd97874b28acfc95cca83801636396c2673703ec800f6675a1b123c3eebb50fe` | `5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc` | slice-8b | INT4_ASYM ratio=1.0 group_size=128 scale_estimation=true dataset=wikitext2; **incomplete** (no awq) | — | `Qwen3-8B-int4-ov.v2.yaml` |
| `Qwen3-8B-int4-ov.v2.yaml` | `cd97874b28acfc95cca83801636396c2673703ec800f6675a1b123c3eebb50fe` | `5c47abf4b8e12ebe8e99745bb0c1ec17e0c0abcc` | slice-8b | INT4_ASYM ratio=1.0 group_size=128 awq=false scale_estimation=true dataset=wikitext2 | `Qwen3-8B-int4-ov.yaml` | — |
| `Qwen3-0.6B-int4-ov.yaml` | `55662b10ef6fced1a39249335130afdb6f8f35a88e7c4ba36cc8c6cde42b8131` | `f864c6106efb6c7f7b4ef274a78a98e37210dddd` | throwaway-install-validator | INT4_ASYM ratio=0.8 group_size=128; **incomplete**. **v1 `verification` is null / incomplete** (no per-file Hub hash gate). Do not use for measurement until re-verified with current `pin_model_ir.py`. | — | `Qwen3-0.6B-int4-ov.v2.yaml` |
| `Qwen3-0.6B-int4-ov.v2.yaml` | `55662b10ef6fced1a39249335130afdb6f8f35a88e7c4ba36cc8c6cde42b8131` | `f864c6106efb6c7f7b4ef274a78a98e37210dddd` | throwaway-install-validator | INT4_ASYM ratio=0.8 group_size=128 awq=null scale_estimation=null dataset=null (card unstated). Recipe complete relative to card; **`--validate-against` v1 exits 2** because v1 `verification` is null — v2 could not be validated for that reason. Needs re-verification before use. | `Qwen3-0.6B-int4-ov.yaml` | — |
| `Qwen3-4B-int8-ov.yaml` | `f7d323d0eb94a6bc9a1bf807416877cbb988cfbd95e7f9ef12403951b90886ae` | `fd9c3cc8adf942151e48904b7341a57eae58d8d5` | slice-4b | INT8_ASYM ratio=null group_size=null awq=false scale_estimation=false dataset=null (**complete v1**; no incomplete ancestor). Confound vs int4: bit width + SYM/ASYM + calibration favour int4 on quality. | — | — |
| `Qwen3-4B-fp16-ov.yaml` | `3dabb40979295a47506bab4073a4d553177d4ec6dc43a93b8a2762dfcd563d1f` | `00298f1c614ffdc000814d8cbe1b7613aa73cdca` | slice-4b | mode=null (uncompressed) ratio=null group_size=null awq=false scale_estimation=false dataset=null (**complete v1**; T2 / W-5 boundary arm). | — | — |

Weight precision is selected by `--model-spec` (or `openvino.model_spec`). KV precision
is selected per arm via `properties.KV_CACHE_PRECISION` in `configs/delta_n.yaml`.
The two are orthogonal.
