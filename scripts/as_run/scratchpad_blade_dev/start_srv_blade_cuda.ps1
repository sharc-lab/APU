$GGUF     = "$env:USERPROFILE\.ollama\models\blobs\sha256-85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
$EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
$SERVER   = "C:\apu\bin\llama-b10970-cuda\llama-server.exe"
$PORT     = 8383
$CTX      = 8192
$LOG      = "C:\apu\logs\blade_cuda_srv.log"

Write-Host "=== start_srv_blade_cuda.ps1 ==="

if (-not (Test-Path $SERVER)) { Write-Error "server not found: $SERVER"; exit 1 }
if (-not (Test-Path $GGUF))   { Write-Error "GGUF not found: $GGUF"; exit 1 }

Write-Host "Computing SHA256..."
$hash = (Get-FileHash $GGUF -Algorithm SHA256).Hash.ToLower()
Write-Host "Computed : $hash"
Write-Host "Expected : $EXPECTED"
if ($hash -ne $EXPECTED) { Write-Error "Hash mismatch. STOP."; exit 1 }
Write-Host "Hash OK."

$existing = Get-Process -Name "llama-server" -ErrorAction SilentlyContinue |
    Where-Object { $_.MainWindowTitle -like "*$PORT*" -or $true }
if ($existing) {
    Write-Host "Stopping existing llama-server PIDs: $($existing.Id -join ', ')"
    $existing | Stop-Process -Force
    Start-Sleep -Milliseconds 500
}

New-Item -ItemType Directory -Force (Split-Path $LOG) | Out-Null

$proc = Start-Process -FilePath $SERVER `
    -ArgumentList "--model `"$GGUF`" --ctx-size $CTX -np 1 --port $PORT" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $LOG `
    -RedirectStandardError "$LOG.err" `
    -PassThru

Write-Host "llama-server (CUDA) PID $($proc.Id)"

$deadline = [DateTime]::UtcNow.AddSeconds(120)
while ([DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Milliseconds 2000
    try {
        $resp = Invoke-RestMethod "http://127.0.0.1:$PORT/health" -ErrorAction Stop
        if ($resp.status -eq "ok") {
            Write-Host "/health OK"
            exit 0
        }
    } catch {}
    Write-Host "  waiting..."
}
Write-Error "Timeout waiting for /health after 120s"
exit 1
