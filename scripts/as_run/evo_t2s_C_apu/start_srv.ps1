Stop-Process -Name llama-server -EA SilentlyContinue
Start-Sleep 1
$exe = "C:\apu\bin\llama-b10970\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$args_list = @("-m", $model, "--port", "8383", "--ctx-size", "8192",
               "--n-gpu-layers", "99", "--no-context-shift",
               "--reasoning-budget", "0", "--reasoning-format", "deepseek",
               "-np", "1", "--log-verbosity", "3")
Start-Process -FilePath $exe -ArgumentList $args_list `
    -RedirectStandardOutput "C:\apu\srv_out3.log" `
    -RedirectStandardError  "C:\apu\srv_err3.log" `
    -WindowStyle Hidden
Start-Sleep 10
$p = Get-Process llama-server -EA SilentlyContinue
if ($p) {
    Write-Host "STARTED PID $($p.Id)"
    try {
        $r = Invoke-WebRequest http://127.0.0.1:8383/health -TimeoutSec 5 -EA Stop
        Write-Host "HEALTH $($r.StatusCode)"
    } catch {
        Write-Host "HEALTH NOT READY YET"
    }
} else {
    Write-Host "ERROR: not started"
}
