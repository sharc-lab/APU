# SEAM project-state MCP

A read-only MCP server that gives every agent one source of truth for project state.

## Why

SEAM is worked by several agents in parallel. Without a shared view, each re-derives
state by reading a dozen files and they can disagree — which already happened once,
when a stale governing-document hash was cited as authoritative.

This server never writes, never mutates, and never touches credentials.

## Design principle

It reports **declared** state (`configs/project_state.yaml`, human-maintained) and
**inferred** state (read from artifacts on disk) separately, and flags disagreement
rather than reconciling it. A divergence means either the declared file is stale or an
artifact is not what it should be — and which one it is matters. That is a human call.

## Install

```powershell
.venv-seam\Scripts\activate
pip install "mcp[cli]" pyyaml
```

Add to `.cursor/mcp.json` at repo root:

```json
{
  "mcpServers": {
    "seam-project-state": {
      "command": "${workspaceFolder}\\.venv-seam\\Scripts\\python.exe",
      "args": ["${workspaceFolder}\\tools\\seam_mcp\\project_state.py"]
    }
  }
}
```

Restart Cursor, then confirm the server appears under Settings → MCP.

## Tools

| Tool | Returns |
|---|---|
| `seam_status()` | Declared vs inferred state, with disagreements listed |
| `seam_runs(kind, target)` | Sealed runs, filterable |
| `seam_run(run_id)` | Full manifest + summary for one run |
| `seam_pins()` | Governing-document hash verdicts (AM-009) |
| `seam_amendments()` | AM ledger with status, and which are open |
| `seam_raw_usage()` | `raw/` size against the AM-014 ceiling |
| `seam_platform()` | Platform A config, plus figures that must not be cited as measured |

## Maintenance

Update `configs/project_state.yaml` in the **same commit** that changes the state it
describes. A stale declared state is worse than none, because it looks authoritative.

`seam_pins()` reports drift but deliberately does not judge whether it is known
supersession or unexplained. That requires reading `AMENDMENTS.md`.
