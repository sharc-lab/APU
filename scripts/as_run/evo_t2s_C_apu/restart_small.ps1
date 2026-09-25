Stop-Process -Name llama-server -EA SilentlyContinue
Start-Sleep 1
$exe = "C:\apu\bin\llama-b10970\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$outlog = "C:\apu\small_out.log"
$errlog = "C:\apu\small_err.log"
$args_list = @("-m", $model, "--port", "8383", "--ctx-size", "8192",
               "--n-gpu-layers", "99", "--no-context-shift",
               "--reasoning-budget", "0", "--reasoning-format", "deepseek",
               "-np", "1", "--log-verbosity", "3")
Start-Process -FilePath $exe -ArgumentList $args_list -RedirectStandardOutput $outlog -RedirectStandardError $errlog -WindowStyle Hidden
Start-Sleep 5
$p = Get-Process llama-server -EA SilentlyContinue
if ($p) { Write-Host "Started PID $($p.Id)" } else { Write-Host "ERROR: not started" }
