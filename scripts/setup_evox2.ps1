<#
.SYNOPSIS
  One-time EVO-X2 (Strix Halo) setup. Run ONCE, at the machine's own keyboard,
  in an elevated (Administrator) PowerShell window.

.PREREQUISITES -- fill these two variables in before running:

  1. $SSH_PUBLIC_KEY  -- the CONTENTS of your SSH public key file (e.g. the
     text inside id_ed25519.pub on the workstation you'll control this
     machine from). This is what lets that workstation SSH in afterward,
     matching the pattern already used for evo-t2s and Blade 14. Paste the
     whole line, starting with "ssh-ed25519 " or "ssh-rsa ".

  2. $TAILSCALE_AUTH_KEY -- a Tailscale auth key tagged tag:apu-x2, generated
     from https://login.tailscale.com/admin/settings/keys ("Generate auth
     key", check "Reusable" if you want to be able to re-run this script,
     and under "Tags" select apu-x2). If tag:apu-x2 does not already exist in
     your tailnet's ACL policy, add it there first (Tailscale admin console
     -> Access Controls -> add "tagOwners": {"tag:apu-x2": ["autogroup:admin"]}
     or similar, scoped to whichever identity owns your tailnet) -- key
     generation will fail or the tag will silently not apply otherwise.

  NOT handled by this script: the model GGUF (qwen3-4b-instruct, sha256
  85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9) is not
  publicly hosted, so it isn't downloaded here. Once this script finishes and
  this machine is reachable over Tailscale, the controlling workstation (or
  Claude, from that workstation) copies it over via a plain scp from evo-t2s
  or Blade 14 -- a single command run from THERE, not here. This script
  creates C:\apu\models and verifies the GGUF's sha256 once that copy lands,
  but does not fetch it itself.

  Also not handled: whether your Windows account on this machine is a local
  Administrator. It needs to be, for OpenSSH Server + Tailscale + system
  Python installs. If you're on a fresh Windows setup you almost certainly
  already are; if not, the OpenSSH and Tailscale install steps will fail
  with an access-denied error and you'll need to re-run elevated.

.WHAT THIS SCRIPT DOES
  1. Enables Windows OpenSSH Server, sets it to start automatically, opens
     the firewall rule, and installs $SSH_PUBLIC_KEY into
     administrators_authorized_keys (assumes an administrator account, the
     Windows OpenSSH convention -- if your account is NOT an administrator,
     it goes to your own .ssh\authorized_keys instead, handled below).
  2. Installs Tailscale (winget if available, else direct MSI download) and
     joins the tailnet with $TAILSCALE_AUTH_KEY, tagged tag:apu-x2.
  3. Installs Python 3.12 (winget, or direct installer download) and pip
     installs numpy, pandas, pytest -- matching what the balloon/contention
     tooling on evo-t2s and Blade 14 already need.
  4. Creates C:\apu, C:\apu\bin, C:\apu\models, C:\apu\results and downloads
     + extracts the llama-b10970 Vulkan Windows build directly from the
     public GitHub release (no private access needed for this part):
     https://github.com/ggml-org/llama.cpp/releases/download/b10970/llama-b10970-bin-win-vulkan-x64.zip
  5. Prints a final status block: what succeeded, what still needs the GGUF
     copied in from elsewhere, and the machine's own Tailscale IP/hostname to
     hand back to whoever is setting up the controlling workstation's access.

  Every step is independently checked and reported; a failure in one step
  does not silently block the others (e.g. if Tailscale auth fails, OpenSSH
  and Python still get set up, and the script tells you exactly what to
  retry and how).
#>

# =====================================================================
# FILL THESE IN BEFORE RUNNING
# =====================================================================
$SSH_PUBLIC_KEY     = "PASTE_YOUR_SSH_PUBLIC_KEY_HERE"
$TAILSCALE_AUTH_KEY = "PASTE_YOUR_TAILSCALE_AUTH_KEY_HERE"
# =====================================================================

$ErrorActionPreference = "Continue"
$results = [ordered]@{}

function Report($step, $ok, $detail) {
    $results[$step] = @{ ok = $ok; detail = $detail }
    $status = if ($ok) { "OK" } else { "FAILED" }
    Write-Host "[$status] $step -- $detail"
}

if (-not (([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))) {
    Write-Host "NOT RUNNING ELEVATED. Re-run this script in an Administrator PowerShell window." -ForegroundColor Red
    exit 1
}

if ($SSH_PUBLIC_KEY -eq "PASTE_YOUR_SSH_PUBLIC_KEY_HERE" -or $TAILSCALE_AUTH_KEY -eq "PASTE_YOUR_TAILSCALE_AUTH_KEY_HERE") {
    Write-Host "Edit this script first: fill in `$SSH_PUBLIC_KEY and `$TAILSCALE_AUTH_KEY at the top." -ForegroundColor Red
    exit 1
}

# --------------------------------------------------------------- 1. OpenSSH
try {
    $capability = Get-WindowsCapability -Online -Name OpenSSH.Server*
    if ($capability.State -ne "Installed") {
        Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0 | Out-Null
    }
    Start-Service sshd
    Set-Service -Name sshd -StartupType Automatic
    if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
    }

    $isAdminAccount = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if ($isAdminAccount) {
        $keyFile = "$env:ProgramData\ssh\administrators_authorized_keys"
    } else {
        $keyFile = "$env:USERPROFILE\.ssh\authorized_keys"
        New-Item -ItemType Directory -Force -Path (Split-Path $keyFile) | Out-Null
    }
    if (-not (Test-Path $keyFile) -or -not (Select-String -Path $keyFile -Pattern ([regex]::Escape($SSH_PUBLIC_KEY)) -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -Path $keyFile -Value $SSH_PUBLIC_KEY
    }
    if ($isAdminAccount) {
        # OpenSSH on Windows refuses administrators_authorized_keys unless its
        # ACL is exactly SYSTEM + Administrators, both Full Control, nothing else.
        icacls $keyFile /inheritance:r | Out-Null
        icacls $keyFile /grant "SYSTEM:F" "Administrators:F" | Out-Null
    }
    Report "OpenSSH Server" $true "sshd running, set to auto-start, firewall rule present, key installed to $keyFile"
} catch {
    Report "OpenSSH Server" $false $_.Exception.Message
}

# --------------------------------------------------------------- 2. Tailscale
try {
    $tsInstalled = Get-Command tailscale -ErrorAction SilentlyContinue
    if (-not $tsInstalled) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Tailscale.Tailscale -e --accept-source-agreements --accept-package-agreements | Out-Null
        } else {
            $msi = "$env:TEMP\tailscale-setup.msi"
            Invoke-WebRequest -Uri "https://pkgs.tailscale.com/stable/tailscale-setup-latest.msi" -OutFile $msi
            Start-Process msiexec.exe -ArgumentList "/i `"$msi`" /quiet /norestart" -Wait
        }
        Start-Sleep -Seconds 5
    }
    $tailscaleExe = "${env:ProgramFiles}\Tailscale\tailscale.exe"
    if (-not (Test-Path $tailscaleExe)) { $tailscaleExe = "tailscale" }  # fall back to PATH
    & $tailscaleExe up --authkey $TAILSCALE_AUTH_KEY --hostname "evox2-strix-halo" --accept-risk=all
    Start-Sleep -Seconds 3
    $tsStatus = & $tailscaleExe status --json | ConvertFrom-Json
    $tsIP = $tsStatus.Self.TailscaleIPs[0]
    Report "Tailscale" $true "joined tailnet, IP=$tsIP, hostname=evox2-strix-halo, tag:apu-x2 (verify tag applied in admin console)"
} catch {
    Report "Tailscale" $false $_.Exception.Message
}

# --------------------------------------------------------------- 3. Python
try {
    $pyCmd = Get-Command python -ErrorAction SilentlyContinue
    $needInstall = $true
    if ($pyCmd) {
        $ver = & python --version 2>&1
        if ($ver -match "3\.12") { $needInstall = $false }
    }
    if ($needInstall) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            winget install --id Python.Python.3.12 -e --accept-source-agreements --accept-package-agreements | Out-Null
        } else {
            $pyInstaller = "$env:TEMP\python312-setup.exe"
            Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe" -OutFile $pyInstaller
            Start-Process $pyInstaller -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1" -Wait
        }
        # Refresh PATH in this session
        $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    }
    python -m pip install --upgrade pip | Out-Null
    python -m pip install numpy pandas pytest | Out-Null
    $verCheck = python -c "import sys,numpy,pandas,pytest; print(sys.version.split()[0], numpy.__version__, pandas.__version__, pytest.__version__)"
    Report "Python 3.12 + numpy/pandas/pytest" $true "versions: $verCheck"
} catch {
    Report "Python 3.12 + numpy/pandas/pytest" $false $_.Exception.Message
}

# --------------------------------------------------------------- 4. C:\apu + Vulkan build
try {
    New-Item -ItemType Directory -Force -Path C:\apu, C:\apu\bin, C:\apu\models, C:\apu\results | Out-Null
    $vulkanUrl = "https://github.com/ggml-org/llama.cpp/releases/download/b10970/llama-b10970-bin-win-vulkan-x64.zip"
    $vulkanZip = "C:\apu\llama-b10970-vulkan.zip"
    Invoke-WebRequest -Uri $vulkanUrl -OutFile $vulkanZip
    Expand-Archive -Path $vulkanZip -DestinationPath "C:\apu\bin\llama-b10970" -Force
    $serverExe = Get-ChildItem -Path "C:\apu\bin\llama-b10970" -Filter "llama-server.exe" -Recurse | Select-Object -First 1
    if (-not $serverExe) { throw "llama-server.exe not found after extraction" }
    Report "llama-b10970 Vulkan build" $true "extracted to $($serverExe.DirectoryName)"
} catch {
    Report "llama-b10970 Vulkan build" $false $_.Exception.Message
}

# --------------------------------------------------------------- 5. GGUF placeholder check
$gguf = "C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
$expectedSha = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
if (Test-Path $gguf) {
    $actualSha = (Get-FileHash $gguf -Algorithm SHA256).Hash.ToLower()
    if ($actualSha -eq $expectedSha) {
        Report "Model GGUF" $true "already present and sha256-verified"
    } else {
        Report "Model GGUF" $false "present but SHA256 MISMATCH (expected $expectedSha, got $actualSha) -- delete and re-copy"
    }
} else {
    Report "Model GGUF" $false "NOT YET PRESENT -- copy from evo-t2s or Blade 14 once this machine is reachable over Tailscale (from the controlling workstation, not from here), then re-run this script to verify its sha256, or verify manually with: Get-FileHash '$gguf' -Algorithm SHA256"
}

# --------------------------------------------------------------- summary
Write-Host "`n============================================================"
Write-Host "SUMMARY"
Write-Host "============================================================"
foreach ($k in $results.Keys) {
    $r = $results[$k]
    $status = if ($r.ok) { "OK" } else { "ACTION NEEDED" }
    Write-Host "$status : $k"
    Write-Host "        $($r.detail)"
}
Write-Host "============================================================"
Write-Host "Hand this machine's Tailscale IP/hostname back to whoever set up your controlling workstation's SSH access."
