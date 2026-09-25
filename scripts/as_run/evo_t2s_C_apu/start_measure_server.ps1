$port = 8383
$exe = "C:\apu\bin\llama-b10970\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$log = "C:\apu\measure_srv.log"

# Kill any leftover server on this port
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Start server (no reasoning budget needed for tokenize; keep it minimal)
Start-Process -FilePath $exe -ArgumentList `
    "-m", $model, `
    "--ctx-size", "512", `
    "--n-gpu-layers", "99", `
    "--port", "$port", `
    "--log-file", $log, `
    "--log-verbosity", "1", `
    "-np", "1", `
    "--no-context-shift", `
    "--reasoning-format", "deepseek" `
    -NoNewWindow -PassThru | Out-Null

Write-Host "Server started on port $port, log at $log"
Write-Host "Waiting for /health..."
$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            Write-Host "Server ready."
            break
        }
    } catch {}
    Start-Sleep -Seconds 2
}
