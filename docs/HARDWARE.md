# Hardware

Per-machine facts that the harness and the analysis scripts need but that are not derivable from a single result row.
For evo-t2s the fuller preflight record is `harness/t2s_preflight.py`'s own output and `configs/hardware/evo_t2s.yaml`;
this file adds only what is new or was found live. For evo-x2 this is the primary record until a preflight script
exists for it.

## evo-t2s (Intel, primary target)

See `configs/hardware/evo_t2s.yaml` and `harness/t2s_preflight.py`. One correction: that yaml's comment calls the CPU
"Arrow Lake"; the actual part (`Intel(R) Core(TM) Ultra X7 358H`) is Panther Lake, confirmed in
`docs/T2S_OVERNIGHT_REPORT.md`. Not fixed in the yaml yet.

## evo-x2 (AMD, primary target)

Set up and verified by the user at the keyboard, 2026-09-28.

| field | value |
|---|---|
| hostname | EVO-X2 |
| hw_id | evo-x2 |
| Tailscale name / IP | evox2-strix-halo / 100.118.33.76 |
| Tailscale tag / ACL | tag:apu-x2; autogroup:member -> tag:apu-x2 port 22 only; autogroup:shared (Zach) has no access |
| local admin account | evo-x2\ritz (local account, not a Microsoft account) |
| SSH | key-only: PasswordAuthentication no, KbdInteractiveAuthentication no, ChallengeResponseAuthentication no; key in C:\ProgramData\ssh\administrators_authorized_keys (ACL SYSTEM + Administrators only); default shell Windows PowerShell 5.1 |
| Windows build | 10.0.26100 (confirmed live via $PSVersionTable.PSVersion: 5.1.26100.7705) |
| Python | %USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe, 3.12.10 (numpy 2.5.3, pandas 3.0.6, pytest, psutil 7.2.2) |
| llama-server | C:\apu\bin\llama-b10970\llama-server.exe, build 10970, commit bfdc32183, Vulkan |
| GPU | AMD Radeon(TM) 8060S Graphics (Strix Halo iGPU), driver 32.0.31007.1017 |
| GPU memory (llama-server --list-devices) | Vulkan0: 98,123 MiB total, 93,217 MiB free |
| GPU memory (dxdiag) | Dedicated 65,360 MB, Shared 32,587 MB |
| RAM (Windows-visible) | 63.6 GB (BIOS reserves 64 GB of physical RAM for the iGPU as a dedicated block; the dxdiag Dedicated figure above is that reservation) |
| C: free | about 1,824 GB |
| Power plan | standby, hibernate and monitor timeouts on AC set to 0; hibernate off |
| SeLockMemoryPrivilege | granted to Ritz, see docs/RAM_CAP_PROTOCOL.md |
| Deploy dir | C:\apu\ovn (created 2026-09-28) |
| Models dir | C:\apu\models |
| Folders present before this project | C:\apu, C:\apu\bin, C:\apu\models, C:\apu\results |

**Memory architecture is a hybrid, not the plain unified pool evo-t2s has.** 64 GB of physical RAM is a fixed
BIOS-level reservation dedicated to the GPU (the dxdiag "Dedicated" figure), separate from the roughly 32 GB of system
RAM the driver can additionally share to the GPU on demand (the dxdiag "Shared" figure) -- a two-tier pool (fixed
dedicated block plus a dynamic shared extension) rather than evo-t2s's single fully dynamic shared pool. This is the
basis for the two-tier budget-crossing design in section 2f below: crossing from the 64 GB dedicated tier into the
shared tier is a distinct, first-class transition to test, separate from the total-budget question near ~91 GB
(93,217 MiB free) that evo-t2s's Section A tested.

**Downloads:** 7 files into C:\apu\models via C:\apu\dl.ps1 (curl -C -, resumable), appending one
"sha256  bytes  filename" line per file to C:\apu\models\sha256.txt, ending with "ALL DONE" when complete. Verified so
far (2026-09-28): `qwen3-4b-instruct-85e4a5b7.gguf` (85e4a5b7...b18b9, 2,497,280,480 bytes) and `Qwen3-8B-Q4_K_M.gguf`
(d98cdcbd...5745785, 5,027,783,488 bytes) match evo-t2s's known hashes exactly; `Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
(7b064f58...033557c, 4,920,739,232 bytes) also matches. Remaining: Qwen3-14B, Qwen3-30B-A3B-Instruct-2507, Qwen3-32B
(Q4_K_M), and Llama-3.3-70B-Instruct-Q4_K_M (bartowski/Llama-3.3-70B-Instruct-GGUF, expected size 42,520,398,816 bytes,
sha256 to be checked against the Hugging Face API once downloaded).

**Known issue at setup time, not caused by this session:** a reboot was already pending
(`HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Component Based Servicing\RebootPending` exists) before any harness
work started. No reboot has been performed; the download window is not to be interrupted and no reboot is to happen
until sha256.txt reads ALL DONE, per standing instruction.

Every Windows-setting change on this machine is logged in `docs/X2_CHANGELOG.md`, with the before value, the after
value, the reason and the revert command.
