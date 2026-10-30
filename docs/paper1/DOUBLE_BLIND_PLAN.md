# Double-blind plan (ISPASS: abstract Dec 11, paper Dec 18)

Written 2026-10-29, read-only. Nothing here changes repository visibility or rewrites history; those are the
operator's and the advisor's decisions. Personal usernames, emails and addresses are referred to by category only,
because this file is itself public. Counts are files in the tree at the time of writing (git grep), not history.

## Why it matters

The repository is public and holds the paper draft, every result the paper cites, and the demo. A reviewer who
searches a distinctive phrase, a number or a model-and-window combination from the paper can find it, and with it
the authors. Double-blind policies usually ask authors not to advertise the work during review and to anonymize the
submission; they rarely require taking existing material down, but the venue's call for papers is the authority
and must be read before Dec 11.

## What identifies the authors today

| category | where | notes |
|---|---|---|
| Repository name and organization | the remote URL (the lab's GitHub organization and the project name); about 18 docs and code files name the organization | The project name also appears in the paper's own tooling (`apu_bench`). |
| Commit author names and emails | every commit in history; author name and personal email of the researcher | Visible to anyone who clones. History is not to be rewritten. |
| Names in docs | the advisor's name and institution in about 6 files (for example `docs/IP_DISCLOSURE_LOG.md`, meeting and outline docs); the researcher's name in commit metadata and some planning docs | Planning docs, not the paper draft. |
| Machine names | the two remote machines' hostnames in about 36 docs files, the device profile `apu_bench/profiles/evo-x2-qwen3-14b.json` ("hardware.hostname"), queue and changelog files; the NVIDIA laptop's model name in about 28 docs files | The paper draft uses platform descriptions (AMD APU, Intel laptop, NVIDIA laptop), not hostnames. |
| Account names and paths | the operator's Windows user name inside home-directory paths in scripts, logs and docs; remote account names in SSH configuration and deploy scripts | `docs/IP_DISCLOSURE_LOG.md` ("Committed personal usernames and paths") records that these were reported to the operator directly. |
| Tailscale addresses | the private overlay addresses of the remote machines in `docs/HARDWARE.md`, `docs/T2S_WEEK_PLAN.md`, `docs/X2_CHANGELOG.md`, `harness/host_config.py`, `scripts/t2s_responsiveness_probe.py` and possibly result logs | Not routable from outside the tailnet, but they tie the files to one account. |
| Demo recordings | `demo/review/` holds 3 screen recordings (.webm) and screenshots, 32 media and PDF files in all | May show desktop user names, window titles, hostnames or browser tabs; each needs a frame-level check. |
| Self-citations | none found in `docs/PAPER1_DRAFT.md` (no prior paper by the authors is cited) | If one is added, cite it in the third person per the venue's rules. |
| The paper's numbers | the register (`docs/NUMBERS_REGISTER.md`), `results/`, `docs/FINDINGS.md` | Exact numbers from the paper are searchable in the public repo. |
| Paper source in the repo | `docs/PAPER1_DRAFT.md`, `docs/paper1/` (tex, PDF) | The draft title and abstract are public now. |

## Options (operator decides; none taken here)

1. Ask the venue chairs whether a public development repository is acceptable during review, with no link from the
   paper. Many venues accept it if the authors do not advertise.
2. Make the repository private for the review period (a visibility change: operator and advisor only). Note that
   forks and clones made earlier stay public, and the IP log's "first public" dates are already fixed.
3. Keep it public but stop adding paper material during review: move the draft and the submission PDF to a private
   location from now on.
4. For any artifact submitted with the paper, build an anonymous copy (see `docs/APU_BENCH_RELEASE_PLAN.md`, stage 1)
   from an allow-list of files, with hostnames, account names, addresses and media removed, in a fresh repository with
   no history.

## Submission checklist (the PDF itself)

- Author block "Anonymous Author(s)" (already so in `docs/paper1/paper1.tex`).
- No repository URL, organization, lab or machine hostname in the text, figures, captions or bibliography.
- PDF metadata (Author, Creator fields) cleared; check with `pdfinfo`.
- Acknowledgments and funding removed for review.
- No register comments printed (they are LaTeX comments; check the PDF text for "reg:"). If LaTeX source is ever
  uploaded, strip the "% reg:" comments first: about 22 of them carry register row ids that contain machine names.
- PDF metadata checked 2026-10-29 on the current build: no Author field; Creator "TeX", Producer MiKTeX pdfTeX.
- Figures re-rendered without hostnames in titles or file paths.
