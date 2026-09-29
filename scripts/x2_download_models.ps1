# Downloads the 7 models this project needs into C:\apu\models on evo-x2, resumable (curl -C -), one line per
# finished file appended to C:\apu\models\sha256.txt, ending with "ALL DONE".
#
# Single-instance: takes a named mutex (Global\apu_dl) and exits immediately if another copy already holds it, so a
# relaunch after a reboot can never run two copies at once. Two copies writing the same files caused a corrupted
# sha256.txt line and a Get-FileHash failure on a locked file on 2026-09-28; this script exists to make that
# structurally impossible rather than relying on the caller not to double-launch.
#
# Stall detection: every curl call also carries --speed-limit 100000 --speed-time 120 --retry 10 --retry-delay 10
# --retry-all-errors, so a transfer that drops below 100 KB/s for 2 minutes straight is killed by curl itself and
# retried, instead of sitting at 0 B/s indefinitely the way the Qwen3-14B download did on 2026-09-28 (stuck at
# exactly 52,879,360 bytes for hours; killing that curl.exe and even a clean process restart did not get it moving
# again, so this flag-level fix -- not a process-level restart -- is what actually addresses that failure mode).
#
# Deployed as C:\apu\dl.ps1 on evo-x2. Relaunch after an interruption with exactly:
#   Start-Process powershell -WindowStyle Minimized -ArgumentList '-NoProfile -ExecutionPolicy Bypass -File C:\apu\dl.ps1'
# (curl -C - resumes partial files; already-finished files are skipped by their own idempotent hash check below).

$ErrorActionPreference = 'Continue'
$mutex = New-Object System.Threading.Mutex($false, "Global\apu_dl")
if (-not $mutex.WaitOne(0)) {
    Write-Output "another copy already holds Global\apu_dl, exiting"
    exit 1
}
try {
    $items = @(
        @('qwen3-4b-instruct-85e4a5b7.gguf', 'https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9', '85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9'),
        @('Qwen3-8B-Q4_K_M.gguf', 'https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf', 'd98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785'),
        @('Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', '7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c'),
        @('Qwen3-14B-Q4_K_M.gguf', 'https://huggingface.co/Qwen/Qwen3-14B-GGUF/resolve/main/Qwen3-14B-Q4_K_M.gguf', '500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0'),
        @('Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', 'https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/main/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf', '6c997b8af17debdfb01d890214400ccbab00db6acc0ba8da5de1cc906c4774d0'),
        @('Qwen3-32B-Q4_K_M.gguf', 'https://huggingface.co/Qwen/Qwen3-32B-GGUF/resolve/main/Qwen3-32B-Q4_K_M.gguf', 'efd971561896866f0e910cce52761ca77b1b138090c7f15fe284676d57d1f689'),
        @('Llama-3.3-70B-Instruct-Q4_K_M.gguf', 'https://huggingface.co/bartowski/Llama-3.3-70B-Instruct-GGUF/resolve/main/Llama-3.3-70B-Instruct-Q4_K_M.gguf', $null)  # checked against the HF API sha256 separately, not hardcoded here
    )
    foreach ($it in $items) {
        $name, $url, $expected = $it[0], $it[1], $it[2]
        $out = 'C:\apu\models\' + $name
        $already = Test-Path $out
        if ($already -and $expected) {
            $h0 = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
            if ($h0 -eq $expected) {
                Write-Output "$name already correct, skipping download"
                continue
            }
        }
        for ($i = 1; $i -le 5; $i++) {
            & curl.exe -L -C - --speed-limit 100000 --speed-time 120 --retry 10 --retry-delay 10 --retry-all-errors -o $out $url
            if ($LASTEXITCODE -eq 0) { break }
        }
        $h = (Get-FileHash $out -Algorithm SHA256).Hash.ToLower()
        Add-Content C:\apu\models\sha256.txt ("{0}  {1}  {2}" -f $h, (Get-Item $out).Length, $name)
    }
    # Dedupe: keep only the last line per filename before the final marker, so a relaunch's repeated lines for an
    # already-finished file do not make sha256.txt ambiguous.
    $lines = Get-Content C:\apu\models\sha256.txt | Where-Object { $_ -ne 'ALL DONE' -and $_.Trim() -ne '' }
    $lastByName = [ordered]@{}
    foreach ($l in $lines) {
        $parts = $l -split '\s\s+'
        if ($parts.Count -ge 3) { $lastByName[$parts[2]] = $l }
    }
    # $lastByName.Values is an OrderedDictionaryKeyValueCollection, not a plain array; passed straight to
    # Set-Content's positional -Value it was NOT enumerated line-by-line and instead got its .ToString() written as
    # one literal line ("System.Collections.Specialized.OrderedDictionary+OrderedDictionaryKeyValueCollection"),
    # silently destroying every hash line while leaving "ALL DONE" intact -- found on 2026-09-29 after the real run
    # finished; all 7 files were downloaded correctly, only this summary file was corrupted, recomputed by hand from
    # the actual files. [string[]] forces real enumeration into a string array first.
    Set-Content C:\apu\models\sha256.txt ([string[]]$lastByName.Values)
    Add-Content C:\apu\models\sha256.txt 'ALL DONE'
}
finally {
    $mutex.ReleaseMutex()
}
