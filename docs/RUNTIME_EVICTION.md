# Runtime Eviction Behaviour — llama-server b10970 and Ollama 0.32.9

**Date characterised:** 2026-09-15  
**Platform:** evo-t2s (Intel Arrow Lake, unified memory), llama-server b10970 Vulkan  
**Model:** Qwen3-4B-Q4_K_M (HF GGUF, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`)

---

## Oversized prompt behaviour (prompt > ctx_size)

### llama-server b10970

Both `--no-context-shift` (default) and `--context-shift` return HTTP 400 with
the following error body when the prompt token count exceeds `n_ctx_slot`:

```json
{
  "error": {
    "code": 400,
    "message": "request (284 tokens) exceeds the available context size (256 tokens), try increasing it",
    "type": "exceed_context_size_error",
    "n_prompt_tokens": 284,
    "n_ctx": 256
  }
}
```

The log emits:

```
E srv    send_error: task id = 0, error: request (284 tokens) exceeds the available context size (256 tokens), try increasing it
I slot      release: id  0 | task 0 | stop processing: n_tokens = 0, truncated = 0
```

**`--context-shift` does not alter this behaviour.** Both configurations reject
the request identically. The `--context-shift` flag has no effect on input
overflow; it applies only to generation overflow (see below).

### Ollama 0.32.9

Also returns HTTP 400 when `num_ctx < prompt_length`. Documented in
`docs/POSITION_PRESSURE.md` T-C01, which is the motivation for the llama-server
backend.

**Neither runtime tested produces silent prompt truncation.** Harness-level
truncation (used in Stage C) models application-layer context management, not
runtime behaviour.

---

## Context-shift flag scope

The `--help` description for `--context-shift` (verbatim from b10970):

```
--context-shift, --no-context-shift     whether to use context shift on infinite text generation (default:
                                        disabled)
                                        (env: LLAMA_ARG_CONTEXT_SHIFT)
```

The phrase "infinite text generation" is the operative scope. Context shift
applies when the cumulative token count (prompt + generated tokens so far)
exceeds `n_ctx_slot` during generation. It does not apply to the initial prompt.

---

## Generation-overflow eviction policy (llama-server b10970)

Measured with `--context-shift --keep 0`, prompt 34 tokens, two `n_ctx_slot`
values. The shift fires when the slot is full and generation needs to continue.

### Verbatim log lines

`n_ctx_slot = 256` (`--ctx-size 256`):

```
0.07.935.631 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
0.11.275.868 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
0.14.604.724 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
```

`n_ctx_slot = 512` (`--ctx-size 512`):

```
0.15.010.139 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 511, n_discard = 255
```

### Policy observed

| `n_ctx_slot` | `n_left` at shift | `n_discard` | shifts in one 600-token run |
|---|---|---|---|
| 256 | 255 | 127 | 3 |
| 512 | 511 | 255 | 1 |

- `n_left = n_ctx_slot − 1` at the moment of shift (slot is full, one slot
  available for next token)
- `n_discard = (n_ctx_slot − 1) / 2`, integer division (approximately half)
- Oldest tokens are discarded first; the shift works from the beginning of the
  sequence toward the current position
- Nothing is pinned (`n_keep = 0` is the default, confirmed from `--help`
  verbatim: "number of tokens to keep from the initial prompt (default: 0, -1 = all)")
- `n_discard` scales linearly with `n_ctx_slot`

### Behaviour without context-shift (`--no-context-shift`, default)

Generation stops when the slot is full. `finish_reason: length`, `truncated = 1`
in the release log. With the same 34-token prompt and max_tokens = 400 at
`n_ctx_slot = 256`, tokens_out was 222 (= 256 − 34 = available generation space).

---

## n_ctx_slot floor — 256-token minimum (llama-server b10970)

The server silently raises `--ctx-size` values below 256 to 256. Confirmed at
`--ctx-size 128` → `n_ctx_slot = 256` in three independent runs with fresh log
files. Values at or above 256 are used as-is (`--ctx-size 256` → `n_ctx_slot =
256`; `--ctx-size 512` → `n_ctx_slot = 512`). No log line announces the
adjustment; it is only observable by comparing the requested value to the
`load_model: initializing, n_slots = N, n_ctx_slot = M` startup log line.

The startup log line form (verbatim):

```
I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 256, kv_unified = 'false'
```

**`n_ctx_slot` is the ground truth for effective context size.** `--ctx-size`
is a request, not a guarantee. See `docs/PROTOCOL.md §4` for the recording
requirement.

---

## Context-shift R1 positive control

`--context-shift` emits no startup log line at any verbosity level (tested at
`-lv 5`, 2,800 log lines, no match for `shift`, `context_shift`, or `keep`).
The only evidence that the flag took effect is the `W slot operator(): slot context shift`
warning during generation. See `docs/PROTOCOL.md §1 R1 exception` for the
required per-session overflow probe.
