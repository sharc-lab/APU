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
