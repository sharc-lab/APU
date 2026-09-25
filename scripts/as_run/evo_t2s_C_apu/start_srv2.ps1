$exe   = "C:\apu\bin\llama-b10970\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$log   = "C:\apu\measure_srv.log"
$port  = 8383

# Kill any existing instance on this port
Get-Process -Name "llama-server" -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1

# Write a tiny launcher bat so the server runs in its own process independent of this PS session
$bat = "C:\apu\run_measure_srv.bat"
Set-Content -Path $bat -Value "@echo off"
Add-Content -Path $bat -Value "start """" /B `"$exe`" -m `"$model`" --ctx-size 512 --n-gpu-layers 99 --port $port --no-context-shift --reasoning-format deepseek > `"$log`" 2>&1"
cmd /c $bat

Write-Host "bat launched, waiting 40s for /health..."
$deadline = (Get-Date).AddSeconds(40)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/health" -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    Start-Sleep -Seconds 3
}
if ($ready) {
    Write-Host "READY"
} else {
    Write-Host "TIMEOUT - log follows:"
    Get-Content $log -ErrorAction SilentlyContinue | Select-Object -First 30
}
