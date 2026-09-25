# Kill any leftover llama-server
Stop-Process -Name llama-server -EA SilentlyContinue

# Start fresh
$exe = "C:\apu\llama-server.exe"
$model = "C:\apu\models\Qwen3-4B-Q4_K_M.gguf"
$log = "C:\apu\srv_restart.log"
$args_str = "-m `"$model`" --port 8383 --ctx-size 65536 --n-gpu-layers 99 --no-context-shift --reasoning-budget 0 --reasoning-format deepseek -np 4 --log-verbosity 3"

# Start detached
$pinfo = New-Object System.Diagnostics.ProcessStartInfo
$pinfo.FileName = $exe
$pinfo.Arguments = $args_str
$pinfo.RedirectStandardOutput = $true
$pinfo.RedirectStandardError = $true
$pinfo.UseShellExecute = $false
$pinfo.CreateNoWindow = $true
$proc = New-Object System.Diagnostics.Process
$proc.StartInfo = $pinfo

# Use Start-Process for detached
Start-Process -FilePath $exe -ArgumentList $args_str -RedirectStandardOutput $log -RedirectStandardError $log -WindowStyle Hidden
Start-Sleep 2
$p = Get-Process llama-server -EA SilentlyContinue
if ($p) { Write-Host "Started PID $($p.Id)" } else { Write-Host "ERROR: process not found" }
