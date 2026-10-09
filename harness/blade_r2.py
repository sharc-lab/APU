"""Blade R2 jobs (night 1: llama3.1:8b validation, real, mitigation; night 2: mechanism; night 3: qwen3:8b validation
and real, only if it fits). docs/BLADE_PLAN.md is the plan.

The session driver, scoring, gates and reports are x2_r2_agent's and x2_r2_mechanism's, imported unchanged; this
file only replaces the evo-x2 specifics: the host check, the Ollama lifecycle (harness/blade_common.LocalOllama, a
hidden local child with the pinned 0.34.4 exe, instead of the WMI launch), the version pin, the qwen3:8b fit decision
and the dry-run budget. The protocol is v1 (call-2 mode "off": tools withheld on a turn's second call), as on evo-x2.

  --mode validation   x2_r2_agent.validation_plan("off", neg_arm=BLADE_NEG_ARM): arm b (negative control and
                      baseline, 3 seeds x 10 turns) at num_ctx BLADE_VALIDATION_CTX = 32768 instead of evo-x2's 131072
                      (operator decision 2026-10-08, register row blade-validation-ctx-cap: the largest prompt+generated
                      call in evo-x2's validation sessions is 17567 tokens, under 30000, so 32768 holds every session
                      untruncated and fits the 8 GB GPU far better); positive control num_ctx 8192 (1 seed x 15 turns);
                      diagnostic arm b with tools on call 2. Gates by x2_r2_agent.evaluate_gates with the same arm.
                      --neg-sessions N runs the negative-control arm on the first N of SEEDS_STRONG instead of 3
                      (night 1 rerun, 2026-10-08: 5 sessions, seeds 20260901-05; harness/blade_r2_gate.py reads it).
  --mode real         --tiers (default: default,4096,32768) x SEEDS_STRONG (5) x 40 turns; refuses per model unless
                      the Blade validation file passes validation_preflight_per_model (--require-validation-gates).
  --mode mitigation   x2_r2_mitigation_v1's design: --client-trim margin=0.05 (x2_r2_client_trim), OLLAMA_DEBUG=1 and
                      the mechanism log parsing, x2_r2_agent.mitigation_plan over the Blade tiers (4096, 8192 and the
                      Blade default resolved to K1's measured num_ctx), 3 seeds x 40 turns. The body is
                      x2_r2_agent.run_mitigation, the same function x2_r2_agent.main --mode mitigation runs on evo-x2;
                      the equivalent x2_r2_agent argv is built and parsed with x2_r2_agent.build_arg_parser() and
                      recorded in run_start, so the Blade run is the evo-x2 command with only the server swapped.
  --mode mechanism    x2_r2_agent.mechanism_plan over --tiers, one session per tier, OLLAMA_DEBUG=1, render-only and
                      llama-tokenize prompt checks (x2_r2_mechanism.MechanismRuntime).

Models: --models (always run) plus --models-if-fit (run only if the K1 summary says the model is fully on the GPU at
every context this job uses: the default context and each num_ctx in FIT_CTXS[mode]; blade_k1.fits_at). The decision
is a record in the output. A job left with no model writes {"record": "skipped", "reason": "does not fit ..."} and
exits 0 (night 3's qwen3:8b jobs when K1 says it does not fit).

Tiers other than x2_r2_agent's fixed ones (e.g. a K1 default of 40960) are registered in x2_r2_agent.ARMS at run time
with the same fields (num_ctx, call-2 mode off).

Exit codes: 0 done (or a clean dry-run stop), 2 error or refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

import blade_common as bc  # noqa: E402
import x2_r2_agent as agent  # noqa: E402
from x2_r2_agent import native_chat_body  # noqa: E402,F401  (x2_r2_mechanism.default_checker looks it up here)

CALL2 = "off"
DEFAULT_TIERS = {"real": "default,4096,32768", "mitigation": "default,4096,8192", "mechanism": "default,32768,16384,8192,4096"}
DEFAULT_CLIENT_TRIM = "margin=0.05"
BLADE_VALIDATION_CTX = 32768          # register row blade-validation-ctx-cap
BLADE_NEG_ARM = f"ollama_ctx_{BLADE_VALIDATION_CTX}_negative_control"
# contexts a model must be fully on the GPU at (K1) to join a job through --models-if-fit
FIT_CTXS = {"validation": ("default", BLADE_VALIDATION_CTX, 8192), "real": None, "mitigation": None, "mechanism": None}


def register_blade_arms():
    """The Blade negative-control arm, in every call-2 mode, with its num_ctx (every r2a_turn row records it as
    num_ctx_requested, plus the loaded context Ollama reports)."""
    for m, sfx in agent.MODE_SUFFIX.items():
        agent.ARMS.setdefault(BLADE_NEG_ARM + sfx, {"num_ctx": BLADE_VALIDATION_CTX, "call2_tools": m != "off",
                                                    "call2_mode": m})


register_blade_arms()
EXIT_OK, EXIT_ERR = 0, 2


def tier_base(tier: str) -> str:
    """'default' -> ollama_default; '40960' -> ollama_ctx_40960, registered in x2_r2_agent.ARMS (all call-2 modes)
    if x2_r2_agent does not define it."""
    if tier == "default":
        return "ollama_default"
    n = int(tier)
    base = f"ollama_ctx_{n}"
    for m, sfx in agent.MODE_SUFFIX.items():
        agent.ARMS.setdefault(base + sfx, {"num_ctx": n, "call2_tools": m != "off", "call2_mode": m})
    return base


def tier_arm(tier: str) -> str:
    return tier_base(tier) + agent.MODE_SUFFIX[CALL2]


def neg_seeds(n: int) -> tuple:
    """The first n of x2_r2_agent.SEEDS_STRONG (20260901..20260905): 3 = evo-x2's validation seeds."""
    if not 1 <= n <= len(agent.SEEDS_STRONG):
        raise SystemExit(f"--neg-sessions must be 1..{len(agent.SEEDS_STRONG)}, got {n}")
    return tuple(agent.SEEDS_STRONG[:n])


def plan_for(mode: str, tiers: list[str], seeds=None, turns=None, neg_sessions=None):
    if mode == "validation":
        plan = agent.validation_plan(CALL2, neg_arm=BLADE_NEG_ARM)
        if neg_sessions:
            # only the gate arm (negative control and baseline) grows; positive control and the diagnostic arm as is
            gate_arm = BLADE_NEG_ARM + agent.MODE_SUFFIX[CALL2]
            plan = [(a, neg_seeds(neg_sessions) if a == gate_arm else s, t) for a, s, t in plan]
        return plan
    if mode == "real":
        return [(tier_arm(t), tuple(seeds or agent.SEEDS_STRONG), turns or agent.STRONG_TURNS) for t in tiers]
    if mode == "mechanism":
        return [(tier_arm(t), (agent.SEEDS[0],), turns or agent.MECH_TURNS) for t in tiers]
    if mode == "mitigation":
        return agent.mitigation_plan(CALL2, tiers=tuple(tier_base(t) for t in tiers), seeds=tuple(seeds or agent.SEEDS),
                                     turns=turns or agent.MITIGATION_TURNS)
    raise ValueError(mode)


def fit_ctxs(mode: str, tiers: list[str]) -> list:
    """The contexts a model must fit at for this job: FIT_CTXS[mode], or the job's own tiers."""
    fixed = FIT_CTXS.get(mode)
    return list(fixed) if fixed else [t if t == "default" else int(t) for t in tiers]


def fit_decision(models_if_fit: list[str], k1_summary_path, ctxs=("default",)) -> dict:
    """{model: {"run": bool, "reason": str, "fits": {ctx: bool|None}}} from the K1 summary: run only if the model is
    fully on the GPU (size_vram >= 99% of size) at every ctx in ctxs. A missing summary, model or ctx means "do not
    run" (unknown is never taken as fitting)."""
    import blade_k1
    out = {}
    summ = None
    if k1_summary_path and Path(k1_summary_path).exists():
        summ = json.loads(Path(k1_summary_path).read_text(encoding="utf-8"))
    for m in models_if_fit:
        e = ((summ or {}).get("models") or {}).get(m)
        if summ is None:
            out[m] = {"run": False, "reason": f"no K1 summary at {k1_summary_path}", "fits": {}}
            continue
        if e is None or e.get("missing"):
            out[m] = {"run": False, "reason": "model not measured by K1 (missing from the store)", "fits": {}}
            continue
        fits = blade_k1.fits_at(e, ctxs)
        bad = [c for c, v in fits.items() if v is not True]
        if bad:
            out[m] = {"run": False, "fits": fits,
                      "reason": f"does not fit: K1 shows {m} not fully on the GPU (or not measured) at ctx {bad} "
                                f"(default ctx {e.get('default_ctx')})"}
        else:
            out[m] = {"run": True, "fits": fits,
                      "reason": f"K1: fully on the GPU at {list(fits)} (default ctx {e.get('default_ctx')})"}
    return out


def resolve_default_tier(tiers: list[str], k1_summary_path, model: str):
    """Mitigation trims to a known window, so "default" becomes the num_ctx K1 measured for `model` (duplicates
    dropped, order kept). Without a K1 value the default tier is dropped and the note says so."""
    out, note = [], None
    default_ctx = None
    if k1_summary_path and Path(k1_summary_path).exists():
        e = (json.loads(Path(k1_summary_path).read_text(encoding="utf-8")).get("models") or {}).get(model) or {}
        default_ctx = e.get("default_ctx")
    for t in tiers:
        if t == "default":
            if default_ctx is None:
                note = f"default tier dropped: no K1 default_ctx for {model} in {k1_summary_path}"
                continue
            t = str(int(default_ctx))
            note = f"default tier = K1 default_ctx {t} for {model}"
        if t not in out:
            out.append(t)
    return out, note


class BladeOllamaRuntime(agent.OllamaRuntime):
    """x2_r2_agent's runtime with recover() restarting the Blade's own local server."""
    local_server = None

    def recover(self):
        import host_config as hc
        if hc.wait_for_ollama_ready(timeout_s=60):
            return True
        return self.local_server.restart() if self.local_server is not None else False


def build_arg_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("validation", "real", "mitigation", "mechanism"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="llama3.1:8b")
    ap.add_argument("--models-if-fit", default="", help="comma-separated; run only if K1 says fully on GPU")
    ap.add_argument("--k1-summary", default=None)
    ap.add_argument("--tiers", default=None, help="comma-separated: default or a num_ctx (real/mitigation/mechanism)")
    ap.add_argument("--seeds", default=None, help="comma-separated override (dry runs)")
    ap.add_argument("--turns", type=int, default=None, help="override (dry runs)")
    ap.add_argument("--neg-sessions", type=int, default=None,
                    help="validation mode: sessions on the negative-control arm (seeds 20260901.., max 5; default 3)")
    ap.add_argument("--client-trim", default=None,
                    help=f"mitigation mode (required there), as x2_r2_agent: e.g. {DEFAULT_CLIENT_TRIM}")
    ap.add_argument("--rules-from", default=None)
    ap.add_argument("--require-validation-gates", action="store_true")
    ap.add_argument("--order", choices=("model_major", "seed_major"), default="seed_major")
    ap.add_argument("--ollama-exe", default=None, help="default: the pinned side-by-side 0.34.4 install")
    ap.add_argument("--dry-run-seconds", type=float, default=None)
    ap.add_argument("--allow-version-mismatch", action="store_true", help="dry runs only")
    return ap


def agent_argv(args, models: list[str]) -> list[str]:
    """The x2_r2_agent command this Blade mitigation run is equivalent to (evo-x2's x2_r2_mitigation_v1 command with
    the Blade's output and models; the plan's tiers come from blade_r2, x2_r2_agent's CLI has no tier flag)."""
    argv = ["--mode", "mitigation", "--client-trim", args.client_trim, "--call2-tools", CALL2, "--out", args.out,
            "--models", ",".join(models), "--order", args.order]
    if args.rules_from:
        argv += ["--rules-from", args.rules_from]
        if args.require_validation_gates:
            argv += ["--require-validation-gates", "--per-model-refusal"]
    return argv


def check_mitigation_args(args, models: list[str]) -> tuple[list[str], dict]:
    """Parses agent_argv() with x2_r2_agent's own parser and the client-trim spec with x2_r2_client_trim, the same two
    checks x2_r2_agent.main makes. Raises SystemExit on a bad argv (argparse's own behaviour)."""
    import x2_r2_client_trim as ct
    if not args.client_trim:
        raise SystemExit("--mode mitigation needs --client-trim margin=<fraction>")
    argv = agent_argv(args, models)
    parsed = agent.build_arg_parser().parse_args(argv)
    spec = ct.parse_client_trim(parsed.client_trim)
    return argv, spec


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.client_trim and args.mode != "mitigation":
        raise SystemExit("--client-trim is only used with --mode mitigation")
    dry = args.dry_run_seconds is not None
    if args.allow_version_mismatch and not dry:
        raise SystemExit("--allow-version-mismatch is for dry runs only")
    out = bc.out_path_ok(args.out, dry)
    clock = bc.DryRunClock(args.dry_run_seconds)
    emit = bc.jsonl_emitter(out, clock)

    def log(msg):
        print(f"[{bc.utc_iso()}] {msg}", flush=True)

    bc.require_blade()
    tiers = [t for t in (args.tiers or DEFAULT_TIERS.get(args.mode, "")).split(",") if t]
    seeds = tuple(int(s) for s in args.seeds.split(",")) if args.seeds else None
    tier_note = None
    if args.mode == "mitigation":
        tiers, tier_note = resolve_default_tier(tiers, args.k1_summary, args.models.split(",")[0])
    if args.neg_sessions is not None and args.mode != "validation":
        raise SystemExit("--neg-sessions is only used with --mode validation")
    plan = plan_for(args.mode, tiers, seeds, args.turns, neg_sessions=args.neg_sessions)
    if dry and args.mode == "validation" and (seeds or args.turns):
        plan = [(a, tuple(seeds or s), args.turns or t) for a, s, t in plan]
    models = [m for m in args.models.split(",") if m]
    fit = fit_decision([m for m in args.models_if_fit.split(",") if m], args.k1_summary, fit_ctxs(args.mode, tiers))
    models += [m for m, d in fit.items() if d["run"]]
    mit_argv = client_trim = None
    if args.mode == "mitigation":
        mit_argv, client_trim = check_mitigation_args(args, models)
    exe = args.ollama_exe or bc.PINNED["ollama_exe"]
    # every Ollama process stopped (confirmed) before the version check: the pinned binary, never a running server
    versions, problems = bc.pinned_versions(exe)
    debug = args.mode in ("mechanism", "mitigation")
    import x2_r2_mechanism as mech
    server = bc.LocalOllama(exe=exe, env=dict(mech.OLLAMA_DEBUG_ENV) if debug else None,
                            log_path=bc.LOG_DIR / f"{out.stem}.ollama_serve.log")
    note, rc = "completed", EXIT_OK
    try:
        emit({"record": "run_start", "job": f"blade_r2_{args.mode}", "mode": args.mode, "argv": sys.argv,
              "dry_run": dry, "versions": versions, "version_problems": problems, "models": models,
              "fit_decision": fit, "tiers": tiers, "tier_note": tier_note, "plan": plan, "call2_mode": CALL2, "call2_tools": False,
              "order": args.order, "max_tokens_per_call": agent.MAX_TOKENS_PER_CALL, "keep_alive": agent.KEEP_ALIVE,
              "host": "blade", "client_trim": client_trim, "x2_r2_agent_equivalent_argv": mit_argv})
        for m, d in fit.items():
            log(f"fit decision {m}: run={d['run']} ({d['reason']})")
        if not models:
            reason = "; ".join(f"{m}: {d['reason']}" for m, d in fit.items()) or "no models given"
            emit({"record": "skipped", "reason": reason if reason.startswith("no models") else
                  f"skipped: does not fit ({reason})"})
            log(f"SKIPPED: {reason}")
            note = "skipped"
            return EXIT_OK
        if problems and not args.allow_version_mismatch:
            emit({"record": "refused", "reasons": problems})
            log(f"refused (versions): {problems}")
            return EXIT_ERR
        rules_in_use = agent.RULE_IDS
        refused = {}
        if args.mode in ("real", "mitigation") and args.rules_from:
            vrows = agent.read_rows(Path(args.rules_from))
            pf = agent.validation_preflight_per_model([(Path(args.rules_from).name, vrows)], CALL2, models,
                                                      neg_arm=BLADE_NEG_ARM)
            rules_in_use, refused = pf["rules_in_use"], pf["refused"]
            for m, reasons in refused.items():
                emit({"record": "refused_model", "model_id": m, "reasons": reasons})
                log(f"refused model {m}: {'; '.join(reasons)}")
            if args.require_validation_gates:
                models = list(pf["models_ok"])
                if not models:
                    emit({"record": "refused", "reasons": [r for rs in refused.values() for r in rs]})
                    return EXIT_ERR
        server.start()
        if not server.wait_ready():
            raise RuntimeError("ollama did not become ready")
        sproblems = bc.server_version_problems(server.server_version())
        if sproblems and not args.allow_version_mismatch:
            emit({"record": "refused", "reasons": sproblems})
            log(f"refused (server version): {sproblems}")
            return EXIT_ERR
        runtime = BladeOllamaRuntime()
        runtime.local_server = server
        avail = runtime.available_models()
        missing = [m for m in models if not any(agent.same_tag(m, a) for a in avail)]
        if missing:
            raise RuntimeError(f"models not in the Ollama store: {missing} (have {avail})")
        emit({"record": "runtime", "ollama_version": runtime.version(), "server_log": str(server.log_path),
              "server_env": server.env, **bc.gpu_memory()})
        if args.mode == "mechanism":
            head = mech.read_log(str(server.log_path), 0)
            if not mech.debug_enabled(head):
                raise RuntimeError("OLLAMA_DEBUG not active in the server log")
            mrt = mech.MechanismRuntime(runtime, str(server.log_path), emit, out.parent / (out.stem + "_logs"))
            agent.run_plan(mrt, plan, models, "mechanism", out, log=log, emit=emit, order=args.order,
                           on_session_start=lambda m, a, s: mrt.begin_session(m, a, s,
                                                                              call2_mode=agent.ARMS[a]["call2_mode"]),
                           on_session_end=mrt.end_session)
            rep = mech.write_report(out, agent.read_rows(out))
            log(json.dumps(rep["verdict"], default=str)[:2000])
        elif args.mode == "mitigation":
            rep = agent.run_mitigation(runtime, str(server.log_path), plan, models, out, emit, log, args.order,
                                       client_trim, rules_in_use)
            log(json.dumps(rep["trim_summary"], default=str)[:2000])
        else:
            agent.run_plan(runtime, plan, models, args.mode, out, log=log, emit=emit, order=args.order)
            rows = agent.read_rows(out)
            if args.mode == "validation":
                report = agent.evaluate_gates(rows, call2_tools=CALL2, neg_arm=BLADE_NEG_ARM)
                log("\n" + agent.format_baseline_markdown(report))
            else:
                sessions = [s for s in agent.completed_sessions(rows) if s["mode"] == "real"]
                report = {"rules_in_use": rules_in_use, "refused_models": refused,
                          "kill_criterion": agent.kill_criterion(sessions, rules_in_use, rows)}
            Path(str(out) + ".report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    except bc.DryRunTimeUp as ex:
        note = f"dry run stopped: {ex}"
        log(note)
    except Exception as ex:
        import traceback
        traceback.print_exc()
        note, rc = f"stopped: {ex!r}"[:400], EXIT_ERR
    finally:
        killed = server.stop()
        with open(out, "a", encoding="utf-8") as f:
            # "run_end" only for a finished real run: validation_preflight reads it as "validation finished"
            rec = "run_end" if (rc == EXIT_OK and not dry and note == "completed") else "run_stopped"
            f.write(json.dumps({"record": rec, "note": note, "ollama_killed": killed, "call2_mode": CALL2,
                                "ts_utc": bc.utc_iso()}, default=str) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
