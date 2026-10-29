# CC-VALID: Qwen3-4B-Instruct-2507 INT4 IR, export plan (prepare only)

Task BO part E, Fri Oct 23 2026. **Nothing is downloaded or exported until Zach approves the download.**
Registered cell: CC_AMEND_2026-10-23b.md A10. Queue: run_queue_xps_cc_v3.sh FROM=95 UNTIL=99 with
VALID_SPEC.

## Settings to reproduce: the sealed Qwen3-4B IR rt_info
- Source: openvino_model.xml blob aab6b029, read in derived/audit/CONFIG_PROVENANCE.md section 3.
- Values: nncf 3.0.0, weight_compression mode int4_sym, group_size 128, ratio 1.0, awq True,
  scale_estimation False, gptq False, backup_mode int8_asym, dataset null (data-free AWQ).

## Source checkpoint (verified 2026-10-23)
- Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554.
- config.json: 36 layers, 8 KV heads, head_dim 128, hidden 2,560, intermediate 9,728, vocab 151,936,
  tied embeddings; max_position_embeddings 262,144; rope_scaling null; rope_theta 5,000,000.
- The shapes equal Qwen3-4B's, so k and a are identical by construction.

## Commands (Mac, after the OK; about 20 GB of disk, about 1-1.5 h)
```
python3.11 -m venv ~/seam_closeout_1013/export_2507/.venv
~/seam_closeout_1013/export_2507/.venv/bin/pip install "openvino==2026.2.1" "openvino-tokenizers==2026.2.1.0" "nncf==3.0.0" "optimum-intel[openvino]==2.0.0" huggingface_hub
~/seam_closeout_1013/export_2507/.venv/bin/huggingface-cli download Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 --local-dir ~/seam_closeout_1013/export_2507/src
~/seam_closeout_1013/export_2507/.venv/bin/optimum-cli export openvino --model ~/seam_closeout_1013/export_2507/src --task text-generation-with-past --weight-format fp16 ~/seam_closeout_1013/export_2507/fp16
~/seam_closeout_1013/export_2507/.venv/bin/python ~/seam_closeout_1013/export_2507/export_2507_ir.py ~/seam_closeout_1013/export_2507/src ~/seam_closeout_1013/export_2507/fp16 ~/seam_closeout_1013/export_2507/Qwen3-4B-Instruct-2507-int4-ov
```
- The last step compresses with NNCF directly, using exactly the values above (data-free AWQ).
- If pip cannot resolve nncf 3.0.0, use nncf==3.2.0 (the repo lock) and record it. The rt_info check
  still gates.
- A one-line optimum-cli alternative (`--weight-format int4 --sym --group-size 128 --ratio 1.0 --awq`
  without `--dataset`) is NOT used. Whether optimum-intel 2.0.0 runs a data-free AWQ there is not
  verified.

## Verification (each must pass before the IR is used)
1. The script prints CONFIG_OK: the shapes above and max_position_embeddings 262,144.
2. The script prints RT_INFO_OK: nncf weight_compression mode / group_size / ratio / awq /
   scale_estimation / gptq / backup_mode equal the sealed values. A mismatch exits 3.
3. IR_SHA256.json lists every file. Claude Code writes
   configs/models/Qwen3-4B-Instruct-2507-int4-ov.yaml with those hashes (a new commit, pushed) before
   any CC-VALID cell.
4. After the XPS v3 queue ends:
   `scp -r ~/seam_closeout_1013/export_2507/Qwen3-4B-Instruct-2507-int4-ov <user>@<host>:C:/Users/<user>/Projects/gnn-hls-accel/models/`
   Then check the openvino_model.bin sha256 on the XPS against the spec.
