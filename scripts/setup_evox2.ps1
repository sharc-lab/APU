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

  The model GGUF (qwen3-4b-instruct, sha256
  85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9) IS fetched
  directly by this script -- the blob is public on the Ollama registry, no
  evo-t2s/Blade transfer needed. See step 5 below.

  Also not handled: whether your Windows account on this machine is a local
  Administrator. It needs to be, for OpenSSH Server + Tailscale + system
  Python installs. If you're on a fresh Windows setup you almost certainly
  already are; if not, the OpenSSH and Tailscale install steps will fail
  with an access-denied error and you'll need to re-run elevated.

.WHAT THIS SCRIPT DOES
  1. Enables Windows OpenSSH Server, sets it to start automatically, opens
     the firewall rule, installs $SSH_PUBLIC_KEY into BOTH
     C:\ProgramData\ssh\administrators_authorized_keys (the file OpenSSH on
     Windows actually consults for an account that is a local Administrator
     -- with icacls locked to exactly SYSTEM + Administrators, Full Control,
     nothing else, which OpenSSH requires or it silently ignores the file)
     AND the account's own ~/.ssh/authorized_keys (redundant fallback --
     harmless if unused, but cheap insurance against a future account-type
     change or a Windows OpenSSH version that behaves differently). Restarts
     sshd so the key takes effect immediately, and prints the exact command
     to test from Blade before you walk away from the keyboard.
  2. Installs Tailscale (winget if available, else direct MSI download) and
     joins the tailnet with $TAILSCALE_AUTH_KEY, tagged tag:apu-x2.
  3. Installs Python 3.12 (winget, or direct installer download) and pip
     installs numpy, pandas, pytest -- matching what the balloon/contention
     tooling on evo-t2s and Blade 14 already need.
  4. Creates C:\apu, C:\apu\bin, C:\apu\models, C:\apu\results and downloads
     + extracts the llama-b10970 Vulkan Windows build directly from the
     public GitHub release (no private access needed for this part):
     https://github.com/ggml-org/llama.cpp/releases/download/b10970/llama-b10970-bin-win-vulkan-x64.zip
  5. Downloads the model GGUF directly via curl.exe -L -C - (resumable) from
     the public Ollama registry blob, verifies its sha256, retries the
     download if the hash doesn't match on the first pass.
  6. Prints the BIOS iGPU memory reservation and Windows-visible RAM --
     Strix Halo's BIOS carves out a chunk of physical RAM for the iGPU before
     Windows ever sees it, so this is part of the real memory constraint
     RAM_CAP_PROTOCOL.md's sweep needs to account for, not just the
     bcdedit truncatememory value.
  7. Prints a final summary: hostname, Tailscale IP, OpenSSH status, Python
     version, llama-server --version, GGUF sha256 match -- paste this back.

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
    Set-Service -Name sshd -StartupType Automatic
    if (-not (Get-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -ErrorAction SilentlyContinue)) {
        New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH Server (sshd)" `
            -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 | Out-Null
    }

    # The account is a local admin, so OpenSSH on Windows consults
    # administrators_authorized_keys specifically (NOT the per-user
    # authorized_keys, which it ignores for admin accounts) -- this is the
    # file that actually matters. Its own ~/.ssh/authorized_keys is written
    # too, purely as cheap redundancy in case of a future account-type
    # change or a different OpenSSH build's behavior; it is not load-bearing
    # today.
    $adminKeyFile = "$env:ProgramData\ssh\administrators_authorized_keys"
    New-Item -ItemType Directory -Force -Path (Split-Path $adminKeyFile) | Out-Null
    if (-not (Test-Path $adminKeyFile) -or -not (Select-String -Path $adminKeyFile -Pattern ([regex]::Escape($SSH_PUBLIC_KEY)) -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -Path $adminKeyFile -Value $SSH_PUBLIC_KEY
    }
    # OpenSSH on Windows silently ignores administrators_authorized_keys
    # unless its ACL is EXACTLY SYSTEM + Administrators, both Full Control,
    # nothing else -- any other principal present (e.g. inherited "Users")
    # makes sshd refuse the whole file with no client-visible error.
    icacls $adminKeyFile /inheritance:r | Out-Null
    icacls $adminKeyFile /grant "SYSTEM:F" "Administrators:F" | Out-Null

    $userKeyFile = "$env:USERPROFILE\.ssh\authorized_keys"
    New-Item -ItemType Directory -Force -Path (Split-Path $userKeyFile) | Out-Null
    if (-not (Test-Path $userKeyFile) -or -not (Select-String -Path $userKeyFile -Pattern ([regex]::Escape($SSH_PUBLIC_KEY)) -Quiet -ErrorAction SilentlyContinue)) {
        Add-Content -Path $userKeyFile -Value $SSH_PUBLIC_KEY
    }

    Restart-Service sshd

    $hn = $env:COMPUTERNAME
    Report "OpenSSH Server" $true ("sshd restarted, set to auto-start, firewall rule present, key installed to " +
        "$adminKeyFile (ACL-locked, load-bearing) and $userKeyFile (redundant). " +
        "Test from Blade BEFORE unplugging this keyboard: ssh $env:USERNAME@$hn `"hostname`"  " +
        "(or use this machine's Tailscale IP from the Tailscale step below once that's confirmed, " +
        "since plain hostname resolution may not work off-tailnet)")
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

# --------------------------------------------------------------- 5. GGUF fetch (public blob)
try {
    $gguf = "C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
    $expectedSha = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
    $blobUrl = "https://registry.ollama.ai/v2/library/qwen3/blobs/sha256:$expectedSha"

    $actualSha = $null
    if (Test-Path $gguf) {
        $actualSha = (Get-FileHash $gguf -Algorithm SHA256).Hash.ToLower()
    }

    $maxAttempts = 5
    $attempt = 0
    while ($actualSha -ne $expectedSha -and $attempt -lt $maxAttempts) {
        $attempt++
        Write-Host "  GGUF fetch attempt $attempt/$maxAttempts (curl -L -C - resumes a partial download)..."
        # -C - : resume from wherever the previous attempt left off (or start
        # fresh if the file doesn't exist yet). -L : follow the registry's
        # redirect to the actual CDN-hosted blob.
        & curl.exe -L -C - -o $gguf $blobUrl
        if (Test-Path $gguf) {
            $actualSha = (Get-FileHash $gguf -Algorithm SHA256).Hash.ToLower()
        }
    }

    if ($actualSha -eq $expectedSha) {
        Report "Model GGUF" $true "downloaded and sha256-verified after $attempt attempt(s): $actualSha"
    } else {
        Report "Model GGUF" $false "sha256 MISMATCH after $maxAttempts attempts (expected $expectedSha, got $actualSha) -- delete $gguf and re-run this script's GGUF step, or investigate a registry/CDN issue"
    }
} catch {
    Report "Model GGUF" $false $_.Exception.Message
}

# --------------------------------------------------------------- 6. iGPU BIOS reservation + visible RAM
try {
    # AdapterRAM (Win32_VideoController) is well known to be unreliable for
    # modern iGPUs with dynamically-shared memory -- it frequently reports 0
    # or a wrapped-around garbage value for >4GB. dxdiag's own report is the
    # more trustworthy cross-driver source for "Dedicated"/"Shared" memory
    # figures, since that's the same data Windows itself surfaces to users.
    $dxFile = "$env:TEMP\dxdiag_evox2.txt"
    Start-Process dxdiag.exe -ArgumentList "/t `"$dxFile`"" -Wait
    Start-Sleep -Seconds 2
    $dxText = Get-Content $dxFile -Raw -ErrorAction SilentlyContinue
    $dedicatedMatch = [regex]::Match($dxText, "Dedicated Memory:\s*(.+)")
    $sharedMatch = [regex]::Match($dxText, "Shared Memory:\s*(.+)")
    $dedicated = if ($dedicatedMatch.Success) { $dedicatedMatch.Groups[1].Value.Trim() } else { "not found in dxdiag output" }
    $shared = if ($sharedMatch.Success) { $sharedMatch.Groups[1].Value.Trim() } else { "not found in dxdiag output" }

    $visibleRamKB = (Get-CimInstance Win32_OperatingSystem).TotalVisibleMemorySize
    $totalPhysMB = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1MB

    $detail = "iGPU dedicated memory: $dedicated | iGPU shared (system) memory: $shared | " +
              "Windows-visible RAM: $([math]::Round($visibleRamKB/1024,1)) MB | " +
              "SMBIOS-reported installed RAM: $([math]::Round($totalPhysMB,1)) MB | " +
              "difference (BIOS/firmware + iGPU carve-out before Windows ever sees it): " +
              "$([math]::Round($totalPhysMB - ($visibleRamKB/1024),1)) MB"
    Report "iGPU reservation + visible RAM" $true $detail
    Write-Host "`nRecord both dedicated and shared iGPU memory figures, plus Windows-visible RAM," -ForegroundColor Yellow
    Write-Host "as fields in every RAM_CAP_PROTOCOL.md sweep row -- the BIOS carve-out is part" -ForegroundColor Yellow
    Write-Host "of the real memory constraint on Strix Halo, not just the bcdedit truncatememory value.`n" -ForegroundColor Yellow
} catch {
    Report "iGPU reservation + visible RAM" $false $_.Exception.Message
}

# --------------------------------------------------------------- summary
Write-Host "`n============================================================"
Write-Host "PER-STEP DETAIL"
Write-Host "============================================================"
foreach ($k in $results.Keys) {
    $r = $results[$k]
    $status = if ($r.ok) { "OK" } else { "ACTION NEEDED" }
    Write-Host "$status : $k"
    Write-Host "        $($r.detail)"
}

Write-Host "`n============================================================"
Write-Host "SUMMARY -- paste this block back"
Write-Host "============================================================"

$hostnameOut = $env:COMPUTERNAME

try {
    $tailscaleExeSum = "${env:ProgramFiles}\Tailscale\tailscale.exe"
    if (-not (Test-Path $tailscaleExeSum)) { $tailscaleExeSum = "tailscale" }
    $tsJson = & $tailscaleExeSum status --json 2>$null | ConvertFrom-Json
    $tsIPOut = if ($tsJson.Self.TailscaleIPs) { $tsJson.Self.TailscaleIPs[0] } else { "NOT CONNECTED" }
} catch {
    $tsIPOut = "NOT AVAILABLE ($($_.Exception.Message))"
}

try {
    $sshSvc = Get-Service sshd -ErrorAction Stop
    $sshStatusOut = "$($sshSvc.Status), StartType=$($sshSvc.StartType)"
} catch {
    $sshStatusOut = "NOT INSTALLED / NOT FOUND"
}

try {
    $pyVerOut = (python --version 2>&1)
} catch {
    $pyVerOut = "NOT AVAILABLE"
}

try {
    $llamaServerExe = Get-ChildItem -Path "C:\apu\bin\llama-b10970" -Filter "llama-server.exe" -Recurse -ErrorAction Stop | Select-Object -First 1
    $llamaVerOut = & $llamaServerExe.FullName --version 2>&1
} catch {
    $llamaVerOut = "NOT AVAILABLE ($($_.Exception.Message))"
}

try {
    $ggufPath = "C:\apu\models\qwen3-4b-instruct-85e4a5b7.gguf"
    $expectedShaSum = "85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9"
    if (Test-Path $ggufPath) {
        $actualShaSum = (Get-FileHash $ggufPath -Algorithm SHA256).Hash.ToLower()
        $shaMatchOut = if ($actualShaSum -eq $expectedShaSum) { "MATCH ($actualShaSum)" } else { "MISMATCH (got $actualShaSum, expected $expectedShaSum)" }
    } else {
        $shaMatchOut = "FILE NOT PRESENT"
    }
} catch {
    $shaMatchOut = "CHECK FAILED ($($_.Exception.Message))"
}

Write-Host "Hostname:              $hostnameOut"
Write-Host "Tailscale IP:          $tsIPOut"
Write-Host "OpenSSH (sshd) status: $sshStatusOut"
Write-Host "Python version:        $pyVerOut"
Write-Host "llama-server --version:"
Write-Host "$llamaVerOut"
Write-Host "GGUF sha256:           $shaMatchOut"
Write-Host "============================================================"
