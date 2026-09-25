$PY  = "python"
$SCR = "C:\apu\APU\harness\fig61_stagec_sweep.py"
$OUT = "C:\apu\sweep_out.log"
$ERR = "C:\apu\sweep_err.log"
Set-Content $OUT ""
Set-Content $ERR ""

$mode = $args[0]
if (-not $mode) { $mode = "--smoke" }

$proc = Start-Process -FilePath $PY -ArgumentList "$SCR $mode" `
    -RedirectStandardOutput $OUT -RedirectStandardError $ERR `
    -WindowStyle Hidden -PassThru
Write-Host "sweep PID $($proc.Id) mode=$mode"
