# Correction of the entry-123 "hang" framing in P10, P10-T2S and P11 (Task BB, 2026-10-21)

This note edits nothing in the files below; it records how their framing is to be read.
- DEMO2_PILOT_P10_HANG123_2026-10-20.md
- DEMO2_PILOT_P10_T2S_2026-10-21.md
- DEMO2_PILOT_P11_HANG_REGIME_2026-10-20.md
- DEMO2_PILOT_P11_VARIANTS_2026-10-21.md

No decision rule, threshold or verdict label changes. The words "hang" and "stall"
change only where they were used as facts. Evidence:
analysis/demo2_pilots/BB/ENTRY123_RECLASSIFIED.md and CONTEXT_OVER_LIMIT.md.

1. "The entry-123 turn-0 stall is 4 of 4 (XPS 841b84e4, 24c1abc0, 027a07b7; T2S
   bbdd1c0d)" (P10-T2S "Why on the T2S"; P11 "Why"): overstated. The records are:
   - 841b84e4: STALL;
   - 24c1abc0, 027a07b7, bbdd1c0d: UNKNOWN (a per-turn wall-clock limit fired,
     with no progress signal);
   - 8af10211 (added): TURN_BUDGET_EXCEEDED_PROGRESSING.
2. In P10 and P10-T2S, "hang" in "a knob avoids the hang if its cell completes
   turn 0 within 600 s" means "turn 0 did not finish within 600 s", not "the
   device stalled".
   - The rule is unchanged. Verdicts are read with that meaning: "C1 (027a07b7)
     does not finish turn 0 within 600 s".
   - On the model's own path turn 0 needs about 26 min on the T2S (DERIVED).
     So under 600 s every P10 cell is expected to "not avoid" unless a knob
     changes the trajectory itself (for example a context cap).
3. P11's premise is withdrawn: that the stall sits at step 0 or 1 of the gold
   path (34,050 tokens), and that a stalling step S exists.
   - L1 (9b4cdefb) completed the gold path.
   - The model's own step 1 is 87,392 tokens (8af10211).
   - P11's M/S inputs stay pinned and unused. P11-B
     (DEMO2_PILOT_P11B_2026-10-21.md) replaces them for this question.
4. The P10-T2S cells C2-C6/C1 at 600 s are superseded for this window by P11-B:
   X4 runs C2 and C3 with the stall watchdog and a 3,600 s budget. C4, C5, C6
   and C1 are not scheduled.
5. "Silent device stall under the shipped default configuration" (AY regime
   name) is not established as a regime. What is established:
   - the default runtime accepts prompts more than 2x past the model's 40,960
     positions with no error or warning;
   - at that size a turn can run for tens of minutes while still progressing.
