@echo off
start "" /B "C:\apu\bin\llama-b10970\llama-server.exe" -m "C:\apu\models\Qwen3-4B-Q4_K_M.gguf" --ctx-size 512 --n-gpu-layers 99 --port 8383 --no-context-shift --reasoning-format deepseek > "C:\apu\measure_srv.log" 2>&1
