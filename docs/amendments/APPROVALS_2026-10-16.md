# Approvals recorded 2026-10-16 (Task AT)

Status: REGISTERED 2026-10-16. Zach approved the items below in chat on
2026-10-16 (ET). This file records the approvals only; it edits no
registered file (their hashes are unchanged). Runners do not read it.

## 1. Registered files approved as committed

| file | sha256 | status |
|---|---|---|
| docs/amendments/DEMO2_AMEND_2026-10-16_TRUE_DEFAULT.md | d76c1b080b8415bcaf8f85c7e071031015666bd3dd9a3e30882a5220a8dc0324 | APPROVED by Zach 2026-10-16 |
| docs/amendments/DEMO2_SEARCH_PHASE_2026-10-16.md | 72b12c1ff4f92171cb9b797c1fc1938b6296a786e4b7733b1903c1391ca21fdf | APPROVED by Zach 2026-10-16 |
| docs/amendments/DEMO2_PILOTS_2026-10-16.md | eac93a00e3a3873d4f696f3a1085b77590f2ba66ea7ed9dc97aecf2ecb1c8491 | APPROVED by Zach 2026-10-16 |

## 2. DEMO-2 cloud ceiling

RAISED from $150 to $275 (APPROVED by Zach 2026-10-16). Basis: arm caps at
2 replicates per arm $216.70 plus pilot caps (P7 + P5) $54.81 = $271.51
(true-default amendment section 7; pilots doc section 4). Per-launch caps
are unchanged.

## 3. Spend sequencing

- P7 (A/A) APPROVED now: caps XPS $18.54, T2S $9.00; total $27.54.
- P5 (OBS_MASK K sweep) waits for P7's result and a separate OK from Zach.
- The replicate count per arm follows P7's registered rule (pilots doc
  section 2): 2 if discordant entries > 12 of 45, else 1.

## 4. DEMO2_HELDOUT_60_B base half: rule R1

APPROVED by Zach 2026-10-16. Base half = 30 multi_turn_base entries of the
W-3 pin that are in no DEMO-1 pin and not in DEMO2_HELDOUT_60 (140
eligible), drawn with a fresh random.Random(20261016) stream, file order
kept (tools/demo2_heldout_b.py --full). Condition: the SEARCH phase never
reads a W-3 / d482c621 per-entry outcome for those 30 ids (listed in the
pin's meta under search_barred_outcomes). The selection used ids only; no
W-3 outcome of the chosen entries was looked at.

Full pin: derived/h1_hybrid/demo2/DEMO2_HELDOUT_60_B.json, sha256
9dfa9280c67ef799034338607ad5bbc34246fe02d21fc5882be2fc5a5cfa61ef (base 30
then the registered long half DEMO2_HELDOUT_60_B_LONG30, sha256
46aa65098a4f704d07cd8037168ad48a04269b6731b0118bfb195a50b032c37c, byte for
byte). Registered in ENTRY_PINS with reserved_for the one SEARCH
confirmation run; the runner refuses it without --confirmation-run; -text
in .gitattributes. Loader check: the same data and loader rebuild
DEMO2_HELDOUT_60 and the long half to their pins before the file is
written.
