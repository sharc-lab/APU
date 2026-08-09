#!/usr/bin/env bash
# setup.sh — install Ollama, pull qwen3:4b at int4, verify it responds.
# Run once before the first sweep: bash scripts/setup.sh
set -euo pipefail

# ---------------------------------------------------------------- Ollama
if ! command -v ollama &>/dev/null; then
    echo "[setup] Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
else
    echo "[setup] Ollama already installed: $(ollama --version 2>&1 | head -1)"
fi

# Start daemon if not already running
if ! curl -sf http://localhost:11434/api/tags &>/dev/null; then
    echo "[setup] Starting Ollama daemon..."
    ollama serve &>/dev/null &
    OLLAMA_PID=$!
    echo "[setup] Daemon PID=$OLLAMA_PID — waiting for it to be ready..."
    for i in $(seq 1 20); do
        sleep 1
        curl -sf http://localhost:11434/api/tags &>/dev/null && break
    done
fi

# ---------------------------------------------------------------- Model
MODEL="qwen3:4b"
echo "[setup] Pulling ${MODEL} (q4_K_M quantization)..."
# Try the explicit int4 tag first, fall back to default (also int4)
ollama pull "${MODEL}:q4_K_M" 2>/dev/null || ollama pull "${MODEL}"

# ---------------------------------------------------------------- Smoke test
echo "[setup] Running smoke test..."
RESPONSE=$(curl -sf http://localhost:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"Reply with only the word ready.\",\"stream\":false,\"options\":{\"num_predict\":5,\"temperature\":0}}" \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('response','').strip())")

echo "[setup] Model response: '${RESPONSE}'"

if [[ -z "${RESPONSE}" ]]; then
    echo "[setup] ERROR: empty response — check that the model downloaded correctly." >&2
    exit 1
fi

echo "[setup] Setup complete. Run the sweep with:"
echo "    py -3.11 -m harness.runner"
