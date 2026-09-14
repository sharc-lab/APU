# ΔN launch shim — phased.
#
# The sequential seam.tools.delta_n driver refuses to start against the revised
# configs/delta_n.yaml (interleaved phases). Current phase is ceiling_a only.
#
# Use tools/ceiling_a.ps1. Do NOT launch arm B / three-arm ΔN from here.

$ErrorActionPreference = "Stop"
Write-Host "REFUSED -- sequential delta_n is not the measurement driver."
Write-Host ""
Write-Host "Phase 3 (ceiling_a) launch from the Mac, Cursor/browsers closed on the XPS:"
Write-Host '  ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Orchestrate"'
Write-Host '  ssh xps "cd C:/Users/zjohn/Projects/gnn-hls-accel; powershell -File tools/ceiling_a.ps1 -Status"'
Write-Host ""
Write-Host "Arm B / three-arm ΔN is a separate dispatch after ceiling_a seals."
exit 1
