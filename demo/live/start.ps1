# Start the live run screen (presenter launch, run by hand in a PowerShell window from the repo root).
#   .\demo\live\start.ps1              # LIVE: local lanes on evo-x2 over SSH, cloud lane live (gated, USD 3 per session)
#   .\demo\live\start.ps1 -Mode replay # REPLAYED: the recorded acceptance run, no network at all
# The API key is read from the user environment into this process only and removed when the screen stops.
# If evo-x2 does not answer, the LIVE screen switches to the recorded run by itself (labelled REPLAYED).
# Then open http://127.0.0.1:8800/ (or the -Port given) in the browser; nothing here opens a browser.
param(
    [ValidateSet('live', 'replay')][string]$Mode = 'live',
    [int]$Port = 8800
)
$ErrorActionPreference = 'Stop'
$recorded = 'results/demo_live_run_20261023.jsonl'
try {
    if ($Mode -eq 'live') {
        $env:ANTHROPIC_API_KEY = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
        $env:CLOUD_CLIENT_HOST_ROLE = 'controller'
        $env:DEMO_CLOUD_LIVE = '1'
        py -3.12 -m demo.live --mode live --port $Port
    } else {
        py -3.12 -m demo.live --mode replay --replay $recorded --port $Port
    }
} finally {
    Remove-Item Env:\ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Remove-Item Env:\DEMO_CLOUD_LIVE -ErrorAction SilentlyContinue
}
