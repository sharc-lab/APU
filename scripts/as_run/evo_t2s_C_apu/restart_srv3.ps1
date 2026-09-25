Stop-Process -Name llama-server -EA SilentlyContinue
$exe = "C:\apu\bin\llama-b10970\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$outlog = "C:\apu\srv_out.log"
$errlog = "C:\apu\srv_err.log"
$args_list = @("-m", $model, "--port", "8383", "--ctx-size", "65536",
               "--n-gpu-layers", "99", "--no-context-shift",
               "--reasoning-budget", "0", "--reasoning-format", "deepseek",
               "-np", "4", "--log-verbosity", "3")
Start-Process -FilePath $exe -ArgumentList $args_list -RedirectStandardOutput $outlog -RedirectStandardError $errlog -WindowStyle Hidden
Start-Sleep 4
$p = Get-Process llama-server -EA SilentlyContinue
if ($p) { Write-Host "Started PID $($p.Id)" } else { Write-Host "ERROR: process not started" }
