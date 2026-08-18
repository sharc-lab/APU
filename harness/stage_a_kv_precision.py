"""Stage A — KV precision check.

Stops the running Ollama server, restarts it with controlled environment
variables, loads qwen3:4b-instruct at num_ctx=32768, records VRAM, and
computes bytes/token. Runs four conditions:
  1. f16  (OLLAMA_KV_CACHE_TYPE=f16,   OLLAMA_FLASH_ATTENTION=1)
  2. q8_0 (OLLAMA_KV_CACHE_TYPE=q8_0,  OLLAMA_FLASH_ATTENTION=1)
  3. q4_0 (OLLAMA_KV_CACHE_TYPE=q4_0,  OLLAMA_FLASH_ATTENTION=1)
  4. f16  (OLLAMA_KV_CACHE_TYPE=f16,   OLLAMA_FLASH_ATTENTION unset)

Writes results/gate1_kv_precision.json.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).parent.parent
RESULTS = REPO / "results"
MODEL = "qwen3:4b-instruct"
HOST = "http://localhost:11434"
NUM_CTX_WEIGHTS = 512
NUM_CTX_KV = 32768

# Qwen3-4B architectural constants
# 36 layers, 8 KV heads, 128 head_dim, K+V, bytes-per-element
LAYERS = 36
KV_HEADS = 8
HEAD_DIM = 128
F16_BPT  = LAYERS * KV_HEADS * HEAD_DIM * 2 * 2   # 147,456
Q8_BPT   = LAYERS * KV_HEADS * HEAD_DIM * 2 * 1   #  73,728
Q4_BPT   = LAYERS * KV_HEADS * HEAD_DIM * 2 * 0.5 #  36,864

OLLAMA_EXE = r"C:\Users\rithw\AppData\Local\Programs\Ollama\ollama.exe"


def gpu_used_mib() -> int:
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
        text=True
    ).strip()
    return int(out.split()[0])


def wait_for_server(timeout=60) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{HOST}/api/version", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def kill_ollama():
    """Kill all ollama.exe processes."""
    subprocess.run(
        ["taskkill", "/f", "/im", "ollama.exe"],
        capture_output=True, text=True
    )
    time.sleep(3)


def start_server(extra_env: dict[str, str]) -> subprocess.Popen:
    """Start ollama serve with extra environment variables."""
    env = os.environ.copy()
    env.update(extra_env)
    # Ensure no KV type leaks from parent
    for key in ["OLLAMA_KV_CACHE_TYPE", "OLLAMA_FLASH_ATTENTION"]:
        env.pop(key, None)
    env.update(extra_env)

    proc = subprocess.Popen(
        [OLLAMA_EXE, "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return proc


def load_model(num_ctx: int) -> dict:
    """Load model at given num_ctx, return prompt_eval_count."""
    payload = {
        "model": MODEL,
        "prompt": "hi",
        "options": {"num_ctx": num_ctx, "num_predict": 1, "temperature": 0},
        "stream": False,
    }
    r = httpx.post(f"{HOST}/api/generate", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def unload_model():
    httpx.post(
        f"{HOST}/api/generate",
        json={"model": MODEL, "keep_alive": 0},
        timeout=30
    )
    time.sleep(3)


def get_server_log_kv_info() -> dict:
    """Read the most recent KV cache line from server.log."""
    log_path = Path(r"C:\Users\rithw\AppData\Local\Ollama\server.log")
    result = {"kv_line": None, "flash_attn_line": None, "cpu_kv_mib": None,
              "cuda_kv_mib": None, "kv_type_k": None, "kv_type_v": None}
    if not log_path.exists():
        return result
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    # Scan backwards for recent KV info
    for line in reversed(lines):
        if "llama_kv_cache: size" in line and result["kv_line"] is None:
            result["kv_line"] = line.strip()
            # Parse K/V type
            import re
            m = re.search(r"K \((\w+)\).*V \((\w+)\)", line)
            if m:
                result["kv_type_k"] = m.group(1)
                result["kv_type_v"] = m.group(2)
        if "CPU KV buffer size" in line and result["cpu_kv_mib"] is None:
            import re
            m = re.search(r"([\d.]+) MiB", line)
            if m:
                result["cpu_kv_mib"] = float(m.group(1))
        if "CUDA0 KV buffer size" in line and result["cuda_kv_mib"] is None:
            import re
            m = re.search(r"([\d.]+) MiB", line)
            if m:
                result["cuda_kv_mib"] = float(m.group(1))
        if "Flash Attention" in line and result["flash_attn_line"] is None:
            result["flash_attn_line"] = line.strip()
        if all(v is not None for v in result.values()):
            break
    return result


CONDITIONS = [
    {
        "label": "f16_with_flash",
        "kv_type": "f16",
        "flash_attention": "1",
        "env": {"OLLAMA_KV_CACHE_TYPE": "f16", "OLLAMA_FLASH_ATTENTION": "1"},
    },
    {
        "label": "q8_0_with_flash",
        "kv_type": "q8_0",
        "flash_attention": "1",
        "env": {"OLLAMA_KV_CACHE_TYPE": "q8_0", "OLLAMA_FLASH_ATTENTION": "1"},
    },
    {
        "label": "q4_0_with_flash",
        "kv_type": "q4_0",
        "flash_attention": "1",
        "env": {"OLLAMA_KV_CACHE_TYPE": "q4_0", "OLLAMA_FLASH_ATTENTION": "1"},
    },
    {
        "label": "f16_no_flash",
        "kv_type": "f16",
        "flash_attention": "unset",
        "env": {"OLLAMA_KV_CACHE_TYPE": "f16"},
    },
]


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)

    print("Stage A — KV Precision Check")
    print(f"Model : {MODEL}")
    print(f"num_ctx for weights : {NUM_CTX_WEIGHTS}")
    print(f"num_ctx for KV test : {NUM_CTX_KV}")
    print(f"Architectural bytes/token: f16={F16_BPT:,.0f}  q8_0={Q8_BPT:,.0f}  q4_0={Q4_BPT:,.0f}")
    print()

    output = {
        "ollama_cli_version": "0.32.9",
        "ollama_server_version": "0.32.6",
        "model": MODEL,
        "num_ctx_weights": NUM_CTX_WEIGHTS,
        "num_ctx_kv": NUM_CTX_KV,
        "architectural_bpt": {"f16": F16_BPT, "q8_0": Q8_BPT, "q4_0": Q4_BPT},
        "server_startup_env": {
            "OLLAMA_FLASH_ATTENTION": "false",
            "OLLAMA_KV_CACHE_TYPE": "",
            "note": "Read from server.log startup block"
        },
        "note_flash_attn": (
            "Server started with OLLAMA_FLASH_ATTENTION=false but llama-server "
            "is invoked with --flash-attn auto, which enables Flash Attention "
            "regardless. Flash Attention is active on this build."
        ),
        "weight_vram_mib": None,
        "conditions": [],
    }

    # ── Step 1: measure weight footprint ──────────────────────────────
    print("Step 1: Measuring weight footprint at num_ctx=512")
    print("  Killing any running ollama processes...")
    kill_ollama()

    print("  Starting server (default env)...")
    srv = start_server({"OLLAMA_FLASH_ATTENTION": "1"})
    if not wait_for_server(90):
        print("  ERROR: server did not start. Aborting.")
        srv.kill()
        sys.exit(1)

    print("  Loading model at num_ctx=512...")
    try:
        load_model(NUM_CTX_WEIGHTS)
        time.sleep(2)
        weight_vram = gpu_used_mib()
        output["weight_vram_mib"] = weight_vram
        print(f"  Weight VRAM: {weight_vram} MiB")
    except Exception as e:
        print(f"  ERROR: {e}")
        srv.kill()
        sys.exit(1)

    unload_model()
    srv.kill()
    time.sleep(4)

    # ── Steps 2-5: precision conditions ───────────────────────────────
    for cond in CONDITIONS:
        print(f"\nCondition: {cond['label']}  (kv_type={cond['kv_type']}, flash={cond['flash_attention']})")
        kill_ollama()
        time.sleep(2)

        print("  Starting server...")
        srv = start_server(cond["env"])
        if not wait_for_server(90):
            print("  ERROR: server did not start for this condition. Skipping.")
            srv.kill()
            output["conditions"].append({
                "label": cond["label"], "kv_type": cond["kv_type"],
                "flash_attention": cond["flash_attention"],
                "error": "server_start_timeout"
            })
            continue

        print(f"  Loading model at num_ctx={NUM_CTX_KV}...")
        try:
            load_model(NUM_CTX_KV)
            time.sleep(3)
            total_vram = gpu_used_mib()
            kv_vram = total_vram - weight_vram
            log_info = get_server_log_kv_info()

            # Use log-reported GPU KV if available (more accurate than subtraction)
            if log_info["cuda_kv_mib"] is not None:
                gpu_kv_mib = log_info["cuda_kv_mib"]
            else:
                gpu_kv_mib = kv_vram

            gpu_kv_bytes = gpu_kv_mib * 1024 * 1024
            bpt_measured = gpu_kv_bytes / NUM_CTX_KV
            arch_bpt = {"f16": F16_BPT, "q8_0": Q8_BPT, "q4_0": Q4_BPT}.get(cond["kv_type"], F16_BPT)
            ratio = bpt_measured / arch_bpt if arch_bpt else 0

            result = {
                "label": cond["label"],
                "kv_type_requested": cond["kv_type"],
                "flash_attention": cond["flash_attention"],
                "env": cond["env"],
                "total_vram_mib": total_vram,
                "kv_vram_mib_by_subtraction": kv_vram,
                "gpu_kv_mib_from_log": log_info.get("cuda_kv_mib"),
                "cpu_kv_mib_from_log": log_info.get("cpu_kv_mib"),
                "kv_type_k_from_log": log_info.get("kv_type_k"),
                "kv_type_v_from_log": log_info.get("kv_type_v"),
                "flash_attn_log": log_info.get("flash_attn_line"),
                "gpu_kv_bytes": int(gpu_kv_bytes),
                "bpt_measured": round(bpt_measured, 1),
                "bpt_architectural": arch_bpt,
                "ratio_measured_over_arch": round(ratio, 3),
                "kv_line_from_log": log_info.get("kv_line"),
            }
            output["conditions"].append(result)

            print(f"  Total VRAM: {total_vram} MiB  |  GPU KV: {gpu_kv_mib:.1f} MiB")
            print(f"  KV type from log: K={log_info.get('kv_type_k')} V={log_info.get('kv_type_v')}")
            print(f"  bytes/token: {bpt_measured:,.0f} (arch={arch_bpt:,.0f}, ratio={ratio:.3f})")
            print(f"  Flash Attention: {log_info.get('flash_attn_line', 'not found in log')}")

        except Exception as e:
            print(f"  ERROR: {e}")
            output["conditions"].append({
                "label": cond["label"], "kv_type": cond["kv_type"],
                "flash_attention": cond["flash_attention"],
                "error": str(e)
            })

        unload_model()
        srv.kill()
        time.sleep(4)

    # ── Verdict ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE A VERDICT")
    print("=" * 60)

    valid_conditions = [c for c in output["conditions"] if "error" not in c]
    if len(valid_conditions) >= 3:
        bpts = {c["kv_type_requested"]: c["bpt_measured"] for c in valid_conditions if "bpt_measured" in c}
        f16_bpt = bpts.get("f16", 0)
        q8_bpt  = bpts.get("q8_0", 0)
        q4_bpt  = bpts.get("q4_0", 0)

        if f16_bpt > 0 and q8_bpt > 0 and q4_bpt > 0:
            f16_q8_ratio = f16_bpt / q8_bpt
            f16_q4_ratio = f16_bpt / q4_bpt
            print(f"f16/q8_0 ratio: {f16_q8_ratio:.2f}  (expected ~2.0)")
            print(f"f16/q4_0 ratio: {f16_q4_ratio:.2f}  (expected ~4.0)")

            if 1.7 <= f16_q8_ratio <= 2.3 and 3.4 <= f16_q4_ratio <= 4.6:
                verdict = "PASS — KV quantization is effective. Stage D is unblocked."
            else:
                verdict = "FAIL — Ratios outside 4:2:1 band. KV quantization not taking effect. Stage D CANCELLED."
        else:
            verdict = "FAIL — Insufficient data."
    else:
        verdict = "FAIL — Too many conditions errored."

    output["verdict"] = verdict
    print(verdict)

    # ── Restore server ────────────────────────────────────────────────
    print("\nRestoring server (default env)...")
    kill_ollama()
    srv = start_server({})
    if wait_for_server(60):
        print("Server restored and healthy.")
    else:
        print("WARNING: server did not restart cleanly. Restart Ollama manually.")

    out_path = RESULTS / "gate1_kv_precision.json"
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")
    return verdict


if __name__ == "__main__":
    main()
