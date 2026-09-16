# Runtime Eviction Behaviour — llama-server b10970

**Last updated:** 2026-09-16  
**Test platform:** evo-t2s (Intel Arrow Lake, unified memory), llama-server b10970 Vulkan  
**Model:** Qwen3-4B-Q4_K_M (HF GGUF, SHA-256 `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`)

---

## Scope of this document

This document records what was empirically tested and at what scale. Claims are
restricted to the engine version, endpoints, context sizes, and platform listed
above.

---

## 1. Oversized prompt behaviour — HTTP 400 across all tested scales

### Engine tested

llama-server b10970 Vulkan on evo-t2s. Both the OpenAI-compatible endpoint
(`/v1/chat/completions`) and the native endpoint (`/completion`) were tested.

### Results summary

| ctx_size | n_ctx_slot (logged) | Endpoint | Prompt tokens (actual) | HTTP status | Error type |
|---|---|---|---|---|---|
| 256 | 256 | /v1/chat/completions | 284 | 400 | exceed_context_size_error |
| 8192 | 8192 | /v1/chat/completions | 8614 | 400 | exceed_context_size_error |
| 8192 | 8192 | /completion | 8606 | 400 | exceed_context_size_error |
| 32768 | 32768 | /v1/chat/completions | 34388 | 400 | exceed_context_size_error |
| 32768 | 32768 | /completion | 34380 | 400 | exceed_context_size_error |

All tests used `--no-context-shift` (the b10970 default).

### Verbatim error bodies

ctx_size=256 (from earlier session, reproduced for completeness):
```json
{"error":{"code":400,"message":"request (284 tokens) exceeds the available context size (256 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":284,"n_ctx":256}}
```

ctx_size=8192, /v1/chat/completions:
```json
{"error":{"code":400,"message":"request (8614 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8614,"n_ctx":8192}}
```

ctx_size=8192, /completion (native):
```json
{"error":{"code":400,"message":"request (8606 tokens) exceeds the available context size (8192 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":8606,"n_ctx":8192}}
```

ctx_size=32768, /v1/chat/completions:
```json
{"error":{"code":400,"message":"request (34388 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":34388,"n_ctx":32768}}
```

ctx_size=32768, /completion (native):
```json
{"error":{"code":400,"message":"request (34380 tokens) exceeds the available context size (32768 tokens), try increasing it","type":"exceed_context_size_error","n_prompt_tokens":34380,"n_ctx":32768}}
```

### Verbatim server log lines (ctx=8192, both endpoints)

```
0.02.044.226 E srv    send_error: task id = 0, error: request (8614 tokens) exceeds the available context size (8192 tokens), try increasing it
0.02.044.231 I slot      release: id  0 | task 0 | stop processing: n_tokens = 0, truncated = 0
0.02.068.567 E srv    send_error: task id = 3, error: request (8606 tokens) exceeds the available context size (8192 tokens), try increasing it
0.02.068.574 I slot      release: id  0 | task 3 | stop processing: n_tokens = 0, truncated = 0
```

`truncated = 0` in both log lines: no partial processing occurs before rejection.

### Token count difference between endpoints

`/v1/chat/completions` consistently counts 8 more tokens than `/completion` for
the same input string. The difference is the chat template applied by the compat
layer (role headers, separator tokens). The underlying rejection logic is
identical: both check prompt length against `n_ctx_slot` and reject with
`exceed_context_size_error`.

### Effect of --context-shift

`--context-shift` does not change the rejection behaviour for oversized prompts.
Both `--no-context-shift` (default) and `--context-shift` return HTTP 400 when
the prompt exceeds `n_ctx_slot`. The flag applies only to generation overflow
(see §3).

---

## 2. Ollama — same engine, two access paths

Ollama 0.34.0 on Blade 14 was tested with num_ctx=256 and a ~403-token prompt.

**HTTP status:** 400  
**Full response body (verbatim):**
```json
{"error":"{\"error\":{\"code\":400,\"message\":\"request (403 tokens) exceeds the available context size (256 tokens), try increasing it\",\"type\":\"exceed_context_size_error\",\"n_prompt_tokens\":403,\"n_ctx\":256}}"}
```

The outer JSON has `"error"` as a string containing the inner llama.cpp JSON
object. The inner type is `exceed_context_size_error` — identical to the
llama-server direct response. Ollama bundles llama-server and relays its errors;
it is not an independent runtime. The rejection behaviour is therefore verified
for **one engine (llama.cpp / llama-server b10970) accessed via two paths**:
direct HTTP and via the Ollama wrapper.

When num_ctx=512 and the same 403-token prompt was sent, Ollama returned HTTP
200 with the correct response — confirming that the 400 is governed by `num_ctx`,
not a model-level fixed limit.

**Ollama version tested:** 0.34.0. See THREATS §18 for the version-change
caveat (prior results used 0.32.9).

---

## 3. Context-shift flag scope

The `--help` description for `--context-shift` (verbatim from b10970):

```
--context-shift, --no-context-shift     whether to use context shift on infinite text generation (default:
                                        disabled)
                                        (env: LLAMA_ARG_CONTEXT_SHIFT)
```

"Infinite text generation" is the operative scope. Context shift fires when the
cumulative token count (prompt + tokens generated so far) exceeds `n_ctx_slot`
**during generation**. It does not apply to the initial prompt check.

---

## 4. Generation-overflow eviction policy (llama-server b10970)

Measured with `--context-shift --keep 0`, 34-token prompt, two `n_ctx_slot`
values. The shift fires when the slot is full and generation needs to continue.

### Verbatim log lines

`n_ctx_slot = 256`:
```
0.07.935.631 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
0.11.275.868 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
0.14.604.724 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 255, n_discard = 127
```

`n_ctx_slot = 512`:
```
0.15.010.139 W slot   operator(): id  0 | task 0 | slot context shift, n_keep = 0, n_left = 511, n_discard = 255
```

### Policy observed

| `n_ctx_slot` | `n_left` at shift | `n_discard` | shifts in one 600-token run |
|---|---|---|---|
| 256 | 255 | 127 | 3 |
| 512 | 511 | 255 | 1 |

- `n_left = n_ctx_slot − 1` (slot full, one position reserved for next token)
- `n_discard = (n_ctx_slot − 1) / 2`, integer division
- Oldest tokens discarded first; nothing pinned (`n_keep = 0` is default)
- `n_discard` scales linearly with `n_ctx_slot`

### Without context-shift

Generation stops when the slot is full: `finish_reason: length`, `truncated = 1`
in the log. With a 34-token prompt at `n_ctx_slot = 256`, tokens_out was 222
(= 256 − 34 = available generation budget).

---

## 5. n_ctx_slot floor — 256-token minimum

llama-server b10970 silently raises `--ctx-size` values below 256 to 256.
Confirmed at `--ctx-size 128` → `n_ctx_slot = 256` in three independent runs.
Values at or above 256 are honoured exactly:

| `--ctx-size` requested | `n_ctx_slot` logged |
|---|---|
| 128 | 256 (raised) |
| 256 | 256 |
| 512 | 512 |
| 8192 | 8192 |
| 32768 | 32768 |

No log line announces the adjustment. The startup log form:
```
I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 256, kv_unified = 'false'
```

`n_ctx_slot` is the authoritative effective context size, not `--ctx-size`.
See `docs/PROTOCOL.md §4` for the recording requirement.

---

## 6. Context-shift R1 positive control

`--context-shift` emits no startup log line at any verbosity level (tested at
`-lv 5`, 2,800 log lines; no match for `shift`, `context_shift`, or `keep`).
The only evidence that the flag took effect is the
`W slot operator(): slot context shift` warning during generation. See
`docs/PROTOCOL.md §1 R1 exception` for the required per-session
generation-overflow probe.

---

## 7. No silent prompt truncation

Neither llama-server b10970 nor Ollama (wrapping the same engine) produces
silent prompt truncation at any tested scale (ctx 256 to 32768). Harness-level
truncation used in Stage C models application-layer context management, not
runtime behaviour.
