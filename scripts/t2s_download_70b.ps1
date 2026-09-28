# Downloads Llama-3.3-70B-Instruct-Q4_K_M.gguf into C:\apu\models on evo-t2s, resumable (curl -C -), appends one line
# to C:\apu\models\sha256.txt on completion, ending with "ALL DONE". Mirrors scripts/x2_download_models.ps1's
# single-instance mutex and stall-detection design (see that script's header for the incident that motivated the
# stall flags: a curl transfer stuck at 0 B/s that a plain process restart did not fix).
#
# --limit-rate 30M caps this download's own throughput so it does not compete with night3's co-runner/thermal
# experiments for network or disk bandwidth on the same machine.
#
# Deployed as C:\apu\dl_70b.ps1 on evo-t2s. Relaunch after an interruption with exactly:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\apu\dl_70b.ps1'

$ErrorActionPreference = 'Continue'
$mutex = New-Object System.Threading.Mutex($false, "Global\apu_dl_70b_t2s")
if (-not $mutex.WaitOne(0)) {
    Write-Output "another copy already holds Global\apu_dl_70b_t2s, exiting"
    exit 1
}
try {
    $name = 'Llama-3.3-70B-Instruct-Q4_K_M.gguf'
    $url = 'https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf'
    $expectedSize = 42520398816
    $out = 'C:\apu\models\' + $name
    for ($i = 1; $i -le 10; $i++) {
        & curl.exe -L -C - --speed-limit 100000 --speed-time 120 --retry 10 --retry-delay 10 --retry-all-errors --limit-rate 30M -o $out $url
        if ($LASTEXITCODE -eq 0) { break }
    }
    $len = (Get-Item $out -ErrorAction SilentlyContinue).Length
    if ($len -ne $expectedSize) {
        Write-Output "WARNING: final size $len does not match expected $expectedSize"
    }
    $h = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
    Add-Content C:\apu\models\sha256.txt ("{0}  {1}  {2}" -f $h, $len, $name)
    Add-Content C:\apu\models\sha256.txt 'ALL DONE (70B)'
}
finally {
    $mutex.ReleaseMutex()
}
