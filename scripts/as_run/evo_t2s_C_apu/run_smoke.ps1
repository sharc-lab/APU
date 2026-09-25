hostname
if (-not (Test-Path C:\apu\fig61_sweep.py)) { Write-Host "MISSING fig61_sweep.py"; exit 1 }
py -3 C:\apu\fig61_sweep.py `
    --segments C:\apu\segments.jsonl `
    --scorers  C:\apu\scorers.py `
    --smoke
