# start_srv_instruct.ps1 — launch llama-server with qwen3:4b-instruct GGUF
# NO --reasoning-budget, NO --reasoning-format flags
# ctx=8192, port=8383, single-slot (-np 1)
# Verifies SHA256 before launching. Runs server detached (survives SSH disconnect).

$GGUF     = "C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
$EXPECTED = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
$SERVER   = "C:\apu\bin\llama-server.exe"
$PORT     = 8383
$CTX      = 8192
$OUT_LOG  = "C:\apu\srv_instruct_out.log"
$ERR_LOG  = "C:\apu\srv_instruct_err.log"

Write-Host "=== start_srv_instruct.ps1 ==="
Write-Host "GGUF   : $GGUF"
Write-Host "SERVER : $SERVER"

if (-not (Test-Path $GGUF)) {
    Write-Error "GGUF not found: $GGUF"; exit 1
}
if (-not (Test-Path $SERVER)) {
    Write-Error "llama-server not found: $SERVER"; exit 1
}

Write-Host "Computing SHA256 (may take ~30s for 2.4 GB)..."
$hash = (Get-FileHash -Path $GGUF -Algorithm SHA256).Hash.ToLower()
Write-Host "Computed : $hash"
Write-Host "Expected : $EXPECTED"

if ($hash -ne $EXPECTED) {
    Write-Error "HASH MISMATCH — refusing to start."; exit 1
}
Write-Host "Hash OK."

# Kill any existing llama-server
Stop-Process -Name llama-server -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 2

# Build args array (NO --reasoning-budget, NO --reasoning-format)
$argList = @(
    "--model",    $GGUF,
    "--ctx-size", "$CTX",
    "--port",     "$PORT",
    "-np",        "1",
    "--host",     "127.0.0.1",
    "--log-verbosity", "2"
)

# Clear old logs
Set-Content $OUT_LOG "" -Encoding UTF8
Set-Content $ERR_LOG "" -Encoding UTF8

Write-Host "Launching llama-server detached..."
$proc = Start-Process `
    -FilePath $SERVER `
    -ArgumentList $argList `
    -RedirectStandardOutput $OUT_LOG `
    -RedirectStandardError  $ERR_LOG `
    -WindowStyle Hidden `
    -PassThru

Write-Host "llama-server PID $($proc.Id) started."
Write-Host "Stdout -> $OUT_LOG"
Write-Host "Stderr -> $ERR_LOG"

# Poll /health for up to 120s
Write-Host "Waiting for /health..."
$deadline = (Get-Date).AddSeconds(120)
$ready = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$PORT/health" -TimeoutSec 2 -EA Stop
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch {}
    # Check process still alive
    if (-not (Get-Process -Id $proc.Id -EA SilentlyContinue)) {
        Write-Error "llama-server exited prematurely. Check $ERR_LOG"; exit 1
    }
    Write-Host -NoNewline "."
}

if ($ready) {
    Write-Host "`nSERVER READY on port $PORT"
} else {
    Write-Error "`nTimeout waiting for /health after 120s. Check $ERR_LOG"; exit 1
}
