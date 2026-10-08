# R2 design: two-step agent-session turns

Written 2026-10-06, before any code or run, per instruction. R2 as specified here does not exist yet.
`harness/t2s_r2_session_growth.py` is a different, earlier, single-call-per-turn design (the model gets the
task and answers directly, no tool-call round trip) and is not reused for the turn mechanic below -- it is
reused for session generation (session code, canary, fact seeding, the 5-rule text, the 2-tool schema), which
are pure functions independent of the turn mechanic and do not need rewriting.

## Why two steps, not one

A real agent step is: model sees task + tools -> model emits tool call(s) -> caller runs the tool (here,
simulated) -> caller appends the tool result -> model is called again to produce its actual answer. The old
design scored a single call that mixed "did it call the tool" and "is the final answer correct" into one
pass, which cannot distinguish a model that calls the tool correctly but then answers badly from one that
never calls the tool at all. The two-step design scores each failure mode separately.

## Per-turn mechanic

1. **Call 1 (tool-call turn).** The model receives the user task for this turn, with the 2-tool schema
   (`log_event`, `lookup_fact`) available, exactly as the existing `TOOLS_SCHEMA`/`as_openai_tools()` already
   define them. The model may return tool calls (native `tool_calls`, or a text-embedded call if the native
   field is empty -- same fallback the old design already has, reused as-is).
2. **Scoring call 1:** tool validity and arguments. Two checks, scored independently:
   - **Tool validity**: did the model call a tool that exists in the schema, with syntactically valid
     arguments matching that tool's own parameter schema (right keys, right types)? A call to a
     non-existent tool, or malformed arguments, is a tool-validity failure regardless of which tool was
     intended.
   - **Tool arguments**: for a validly-shaped call, are the argument *values* correct for this turn's task
     (e.g. `lookup_fact(key=...)` using the exact key this turn asked for, not a different fact's key)?
3. **Simulate the tool result.** No real tool runs. `log_event` always simulates success (it has no
   meaningful failure mode in this design -- its only job is rule 2's "was it called at all" check, scored
   in step 5, not here). `lookup_fact` returns the seeded fact value for the requested key, or a deterministic
   "fact not found" string if the model requested a key that was never seeded (a wrong-key call is a tool-
   arguments failure from step 2 and a realistic wrong-result for the model to then recover or not recover
   from in call 2).
4. **Call 2 (final-answer turn).** The same conversation, with the assistant's call-1 message and the
   simulated tool-result message appended, and the model is called again to produce the turn's actual
   answer. This call has tools available too (a real agent step does not forbid a second tool call), but
   this design does not require or score a second round of tool use -- if the model calls a tool again
   instead of answering, that call is still scored for tool validity, and whatever its message ends up being
   is what the text rules in step 5 score, even if it is not the clean JSON answer the rules ask for (that is
   itself a rule-5 failure, correctly captured, not a special case).
5. **Scoring call 2 (text rules, on the FINAL answer only):** the 5 rules already defined in
   `RULE_TEXT_TEMPLATE` -- JSON keys (`answer`/`source`, optional `tool_calls`), metres, session code suffix,
   no literal "ZEBRA-7", plus the canary-repeat check on canary turns (every 5th turn from turn 5, reusing
   `CANARY_CHECK_EVERY`/`generate_canary` as-is). None of these are scored against call 1's own tool-call
   message -- call 1 is not expected to look like a rule-compliant answer at all, since its whole job is to
   emit a tool call, not prose.
6. **The `log_event` rule (rule 2) is scored across BOTH calls of the turn**, not just call 2: it is
   satisfied if `log_event` was called at any point in the turn (call 1's tool call, or a second tool call
   made during call 2). This is deliberately different from the other 4 rules, which only look at call 2's
   final answer -- rule 2 is about agent behavior during the turn, not about the shape of the final text.
7. **Token accounting.** Both calls' prompt and completion tokens count toward the session's cumulative
   token total (the quantity the truncation signal in step 8 is compared against). A two-step turn costs
   roughly double the single-call design's tokens per turn for the same number of turns -- expected and
   intended, not a bug to compensate for by halving turn count.
8. **Truncation detection:** the canary-miss-while-loaded-context-exceeds-window signal, reused as-is from
   the existing design (`generate_canary`/`CANARY_CHECK_EVERY`'s own mechanism: a canary miss while the
   session's real cumulative token count exceeds the runtime's loaded context size is truncation; a canary
   hit, or a miss while still within context, is not). **Never `prompt_eval_count`** as the truncation
   signal -- `prompt_eval_count` reports how much of the *current* call's prompt was processed, not whether
   earlier turns' content silently fell out of the context window several turns ago; the canary is the only
   signal in this design that actually tests recall of content from much earlier in the session.

## Scoring summary per turn

| check | scored against | counts as |
|---|---|---|
| tool validity | call 1 (and call 2, if it also calls a tool) | per-call pass/fail |
| tool arguments | call 1 (and call 2, if it also calls a tool) | per-call pass/fail |
| rule 1 (JSON keys) | call 2's final answer | per-turn pass/fail |
| rule 2 (log_event called) | call 1 OR call 2 | per-turn pass/fail (turn-level OR) |
| rule 3 (no ZEBRA-7) | call 2's final answer | per-turn pass/fail |
| rule 4 (metres) | call 2's final answer | per-turn pass/fail |
| rule 5 (session code suffix) | call 2's final answer | per-turn pass/fail |
| canary (canary turns only) | call 2's final answer | per-turn pass/fail |
| fact recall (recall turns only) | call 2's final answer | per-turn pass/fail |
| truncation | canary miss x loaded-context comparison | per-turn flag, not a rule |

## Validation plan (must pass before any real run)

Arm b only (`num_ctx=131072`), **3 sessions x 10 turns**, two models: **llama3.1:8b** and **qwen3-14b**.

- **Every rule actually exercised in this validation must reach >=90% baseline compliance.** A rule that
  never fires in a 10-turn session (e.g. the canary, which only fires on turn 5 and 10) is reported as
  "not exercised," not silently counted as 100%.
- **Negative control:** 0 canary misses expected at this short length and this context size (10 turns is
  nowhere near exhausting a 131072 window for either model) -- a canary miss here means the harness itself
  is broken (wrong canary text passed, wrong turn comparison, etc.), not a real truncation finding.
- **Positive control:** force an 8192-token effective context (same mechanism the old design's positive
  control already uses) and confirm the canary DOES fire a miss -- if it does not, the canary mechanism
  itself cannot detect truncation and nothing built on top of it can be trusted.
- Report the baseline table: per-rule compliance rate, per model, from this validation run, before any
  real run starts.

## First real run (evo-x2 only, after validation passes)

Three context tiers (the ones already measured in this project): **Ollama default**, **Ollama num_ctx
32768**, **Ollama num_ctx 4096**. 3 seeds, **40 turns**. Same scoring as above. Model(s) for the first real
run: whichever the validation pass confirms both reach >=90% baseline on (llama3.1:8b and qwen3-14b, unless
the baseline table says otherwise).

## What is reused unchanged from `harness/t2s_r2_session_growth.py`

- `generate_session_code`, `generate_canary`, `TOOLS_SCHEMA`, `as_openai_tools()`, `RULE_TEXT_TEMPLATE`,
  `FACT_KEYS`, `CANARY_CHECK_EVERY`, the turn-generation pure functions for task text and recall scheduling.
- The native-tool-call-with-text-fallback detection already built for rule 2's 2026-09-30 fix.

## What is new

- The two-call-per-turn driver (call 1 -> simulate tool result -> call 2).
- Tool-validity and tool-argument scoring as two separate checks (the old design did not separate these).
- Rule 2 scored as a turn-level OR across both calls, instead of against a single call.
- Per-turn token accounting across both calls.
- The validation harness itself (baseline table, negative control, positive control) as a distinct,
  runnable mode before any real-run data is collected.

## Implementation notes (2026-10-06, `harness/x2_r2_agent.py`)

Written with the harness, before its validation run. These pin down choices the sections above leave open; where
one extends the spec it says so.

- **Two canaries (extension).** Ollama's `/api/chat` history truncation keeps every system message and drops the
  oldest non-system messages first, so a canary that lives only in the system prompt can survive a real truncation,
  and the positive control could then never fire. The system-prompt canary (`generate_canary`, unchanged) is kept
  and scored, and a second canary (a "history tag", `generate_history_canary`) is placed in the first user message
  of the session, which is what Ollama drops first. Every canary turn asks for both; both are scored separately; a
  miss of either while over the loaded window is truncation. Which one survives is recorded.
- **Truncation comparator.** "Cumulative token count" is the transcript the window has to hold: the prompt of the
  call that produced the final answer (system prompt, every earlier turn's user message, call-1 tool-call message,
  tool results and call-2 answer, plus the tool schema), estimated at chars/4 and scaled by a per-session
  calibration ratio. The ratio is the turn-1 call-1 `prompt_eval_count` over the estimate, taken right after an
  explicit model unload (nothing cached, far below any window), bounded to [0.5, 2.5], and otherwise unused.
  `prompt_eval_count` is never the truncation signal. The loaded context is `GET /api/ps` `context_length`, read
  after every turn; unknown means no truncation verdict.
- **Both calls' tokens.** The transcript keeps the whole tool round trip of every turn (not only the final answer),
  so later turns pay for earlier turns' tool calls. Each turn row also records the sum of both calls' prompt and
  completion tokens, and the session's running sum of that (a cost figure, not the truncation comparator).
- **No tool call in call 1.** There is nothing to simulate, so there is no call 2: call 1's content is the final
  answer, rule 2 fails, and the row records `n_calls = 1`.
- **Tool calls in call 2.** Scored for validity and arguments and counted for rule 2; their simulated results are
  appended so the transcript stays well formed; no third call is made.
- **Text-embedded calls.** Only when the native `tool_calls` field is empty: the whole content (one markdown code
  fence allowed) is a bare `{"name", "arguments"|"parameters"}` object, a list of them, or a `{"tool_calls": [...]}`
  object without an `"answer"` key. An answer-shaped JSON's optional `"tool_calls"` key is a report, never a call.
- **Tool validity:** known tool, arguments an object with exactly the schema's keys, string values. **Tool
  arguments** (valid calls only): `lookup_fact` is correct only on a lookup turn with exactly that turn's key;
  `log_event` on a length turn must mention the turn's rack id and its length (cm figure, or the same length in
  metres); `log_event` on other turns needs a non-empty event.
- **Simulated `lookup_fact` store** holds one deterministic value per lookup key a turn asks for. The three
  system-prompt recall facts are not in it (looking one up returns "fact not found"), so recall can only come from
  context. Whether the model tried to look a recall key up is recorded.
- **Rule scoring details.** Rule 1: the whole content (one code fence allowed) parses as a JSON object with
  `answer` (string) and `source`, and no keys beyond `answer`/`source`/`tool_calls`. Rule 4 applies to length turns
  only (other turns are excluded from its denominator, not counted as passes) and is scored on the answer field, or
  the raw text if the JSON did not parse, so a rule-1 failure does not cascade into rule 4. Rule 5 needs a parsed
  answer field by definition. Canaries and recall are verbatim substrings of the raw final text.
- **think.** `"think": false` is sent on every qwen3 call and omitted for llama3.1 (no thinking mode); the value is
  recorded per row, and empty final answers are counted per model.
- **Call-2 tools diagnostic.** Step 4 keeps tools available on call 2. Validation also runs arm b with tools
  withheld on call 2 only (`ollama_ctx_131072_call2_notools`, same seeds and turns). Gates are computed on the spec
  arm only; the diagnostic exists so that, if a model loops on tool calls in call 2 instead of answering, the
  alternative's baseline is already measured in the same job.
- **Kill criterion** (pre-registered 2026-09-29, `docs/FINDINGS.md`) is evaluated with
  `t2s_r2_session_growth.evaluate_kill_criterion`. A session's first failure is its first turn with a failed rule
  among the rules in use (those that passed the validation gate), a tool-validity failure, a tool-argument failure,
  or a recall failure; an error is any call with a non-200 status, a transport error, or an error field.

### Changes after the first validation run (2026-10-07, `results/x2_r2_validation.jsonl`)

- **One canary pair per check, never reused.** The first run asked for the same two canaries at every check.
  qwen3:14b's positive control (num_ctx 8192) reproduced both at turn 10 although its final-call prompt (about
  15.5K estimated tokens) was far past the window: the model copied them from its own turn-5 answer, which was still
  inside the retained tail. The system prompt now lists C1..C8 (replacing the single-canary paragraph) and the
  first user message lists H1..H8; turn 5k asks for Ck and Hk only, so no earlier answer can carry the requested
  canary forward. An answer that contains a different check's canary is recorded (`other_canaries_in_answer`).
- **Tools withheld on call 2 for the real run (deviation from step 4).** With tools available on call 2,
  llama3.1:8b answered call 2 with another tool call on 20 of 30 turns in the first run, sometimes stuffing its
  JSON answer into the tool arguments, leaving those 20 final answers empty, so the text rules failed for a harness
  reason, not a context reason. With tools withheld on call 2 only (the diagnostic arm, same seeds) all 30 final
  answers were non-empty. (In the second run, whose prompts differ by the numbered canaries, the spec-variant
  diagnostic arm made no call-2 tool calls, but llama3.1:8b's rule 5 fell to 0% there against 60% with tools
  withheld; qwen3:14b passed every rule under both variants.) Both models therefore run the `_call2_notools` arms
  (`--call2-tools off`), validated again under that variant before the real run. Rule 2 is still a turn-level OR,
  now in practice satisfied only by call 1.
- **Per-model rules in use.** A rule that misses the 90% gate for one model is dropped for that model only;
  `--rules-from <validation file>` makes the real run's kill criterion use each model's gate-passing rules.
- **Positive control** runs 15 turns (two checks past the 8192 window instead of one).

### Call-2 protocol v3 (2026-10-07, operator rule for all R2 runs from here on)

Withholding tools on call 2 (`--call2-tools off`, used by `x2_r2_validation_v2` and `x2_r2_real_v1`) changes the
request between a turn's two calls: the tool definitions are part of the rendered prompt, so call 2 saw a different
prompt prefix than call 1. The rule now is: **identical system prompt and identical tool definitions on every call**,
and the text answer on a turn's second call is forced with `tool_choice: "none"` (or the runtime's equivalent)
instead of by removing tools. `x2_r2_validation_v2` and `x2_r2_real_v1` stay reproducible with `--call2-tools off`.

**What the installed runtime supports.** evo-x2 runs Ollama 0.34.4 (`ollama --version`, 2026-10-07). From the
source at tag v0.34.4:

- `/api/chat`: `api.ChatRequest` (`api/types.go`) has no tool_choice field (model, messages, stream, format,
  keep_alive, tools, options, think, truncate, shift, `_debug_render_only`, logprobs, top_logprobs). The body is bound
  with Go's JSON decoding, which drops unknown keys, so `"tool_choice": "none"` is accepted and has no effect.
  `server/routes.go` ChatHandler has no tool_choice handling either.
- `/v1/chat/completions`: `docs/api/openai-compatibility.mdx` lists `tool_choice` as unsupported (unchecked), and
  `openai.ChatCompletionRequest` has no such field; the middleware binds with `ShouldBindJSON`, so it is dropped the
  same way. This endpoint also cannot set `num_ctx`, which the context tiers need, so it is not usable for R2 runs.
- The one request field that does constrain call 2 to a text answer while the tools stay defined is `format` with a
  JSON schema (grammar-constrained output; ChatHandler passes format and tools independently, and tool-call parsing
  still runs). Its cost: the schema is the answer object, so rule 1 (JSON keys) becomes close to automatic and stops
  measuring instruction following.

So Ollama 0.34.4 has no equivalent of `tool_choice: "none"`. Which way R2 goes is the operator's decision; the
options, all of which keep the system prompt and tools byte-identical on every call:

1. `forced_none` (implemented, queued as the default for v3): send `tool_choice: "none"` on call 2 anyway, which
   on 0.34.4 is the same as tools-on-call-2 (the original step 4), and score any call-2 tool call as a **protocol
   violation** (a turn failure, counted per session and per model). Honest about the rule, but the runtime does not
   enforce it, and the first validation run saw llama3.1:8b answer call 2 with a tool call on 20 of 30 turns.
2. `forced_none_format` (implemented, not queued): option 1 plus `format` = the answer JSON schema on call 2. Enforces
   a text answer; rule 1 must then be dropped from the rules in use (it no longer tests anything).
3. A runtime that implements `tool_choice` (for example llama-server with its chat-template tool support), which
   moves R2 off Ollama and changes what the context tiers mean.

A short queued job, `x2_r2_toolchoice_check` (`--mode toolchoice_check`, file `results/x2_r2_toolchoice_check.jsonl`),
checks this live per model on the real call-2 contexts of a 6-turn forced_none session: the rendered prompt with and
without tool_choice (`_debug_render_only`, compared byte for byte), plain call 2 twice (run-to-run nondeterminism),
call 2 with tool_choice none, call 2 with the format schema, and the same contexts on `/v1/chat/completions` with and
without tool_choice. Its `.report.json` gives per-variant call-2 tool-call counts and a verdict per mechanism.

**What every row records.** Each call: SHA-256 of its system prompt and of its tools JSON (as serialized into the
request), whether tools were sent, the tool_choice sent, and a hash of any format schema. Each turn: `call2_mode`,
`call2_tool_choice_sent`, `system_prompt_identical_across_calls`, `tools_identical_across_calls`, and
`call2_protocol_violation` (forced modes only; a call-2 tool call is still scored for validity, arguments and rule 2
as before). Sessions and the baseline table count protocol violations and turns whose prompt or tools changed.

**Validation v3 and the real-run start check.** `x2_r2_validation_v3` uses the same plan as v2 under forced_none:
arm b (num_ctx 131072) 3 seeds x 10 turns as negative control and baseline, positive control at num_ctx 8192 (1 seed x
15 turns), both models, `think: false` sent and recorded for qwen3:14b, plus the `on` arm b as a diagnostic (the same
request minus tool_choice, a direct check that the field changes nothing). `x2_r2_real_v2` (default vs 32768 vs 4096,
3 seeds x 40 turns, `--rules-from` the v3 file) is gated in the queue on v3 being done, and runs with
`--require-validation-gates`: before starting Ollama it checks the v3 file and refuses to start (queue advanced with a
"refused to start" note, a `refused` record in its output) unless, for each model, validation finished under the same
call-2 mode, both control arms completed, at least one rule is in use and every rule in use is at or above 90%, the
negative control had no canary miss, and the positive control fired on every session.
