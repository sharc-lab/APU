# Cover note for the advisor (draft v1)

Not anonymized; not part of the paper. One paragraph for the operator to send with `PAPER1_DRAFT_v1.pdf`.

---

Dear Prof. Hao, attached is the first typeset draft of the ISPASS 2027 paper on local LLM runtimes that silently drop
conversation history and guardrail rules once the context window is exceeded (IEEE conference format, double-blind,
about 9.6 pages of body plus references; figures are still framed placeholders sized to their specs). Every measured number in
it comes from a verified row of our results register, and the parts still pending are named in the text. The open
items are: the NVIDIA memory panel of the headline figure (the decode-slowdown sweep and the forced-failure point with
the driver's system-memory fallback disabled), which needs a run on the discrete-GPU laptop with me present; the
context-policy and router comparison on the Intel laptop's CPU and integrated GPU, scheduled for the 2026-10-26 to
2026-11-08 access window, since so far it has run only on the AMD APU, whose default window is the model's native one;
larger models on the Intel platform, which I propose to leave as future work; the second model for the tools-on-every-call
task check (mistral-small3.2:24b), whose validation and real-run results have not landed yet, so that check still
rests on qwen3:14b alone; and the qwen3:32b spill sweep on the AMD APU, still pending, so the paper reports that
model's spill onset only as "above 20480 tokens". The fixes section will also get one more update when the measured
comparison of late cloud escalation against routing at session start lands. I would be grateful for your comments on
the framing and on what to cut if the pending figures push the body past ten pages.

Best regards,
Rithwik Sharma
