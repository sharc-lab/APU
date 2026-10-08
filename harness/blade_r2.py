"""Blade R2 jobs (night 1: validation, real, mitigation; night 2: mechanism). docs/BLADE_PLAN.md is the plan.

The session driver, scoring, gates and reports are x2_r2_agent's and x2_r2_mechanism's, imported unchanged; this
file only replaces the evo-x2 specifics: the host check, the Ollama lifecycle (harness/blade_common.LocalOllama, a
hidden local child with the pinned 0.34.4 exe, instead of the WMI launch), the version pin, the qwen3:8b fit decision
and the dry-run budget. The protocol is v1 (call-2 mode "off": tools withheld on a turn's second call), as on evo-x2.

  --mode validation   x2_r2_agent.validation_plan("off"): arm b num_ctx 131072 (3 seeds x 10 turns, negative control
                      and baseline), positive control num_ctx 8192 (1 seed x 15 turns), diagnostic arm b with tools on
                      call 2. Gates by x2_r2_agent.evaluate_gates.
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
its default context: blade_k1.summarize()'s fits_8gb_at_default). The decision is a record in the output.

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


def plan_for(mode: str, tiers: list[str], seeds=None, turns=None):
    if mode == "validation":
        return agent.validation_plan(CALL2)
    if mode == "real":
        return [(tier_arm(t), tuple(seeds or agent.SEEDS_STRONG), turns or agent.STRONG_TURNS) for t in tiers]
    if mode == "mechanism":
        return [(tier_arm(t), (agent.SEEDS[0],), turns or agent.MECH_TURNS) for t in tiers]
    if mode == "mitigation":
        return agent.mitigation_plan(CALL2, tiers=tuple(tier_base(t) for t in tiers), seeds=tuple(seeds or agent.SEEDS),
                                     turns=turns or agent.MITIGATION_TURNS)
    raise ValueError(mode)


def fit_decision(models_if_fit: list[str], k1_summary_path) -> dict:
    """{model: {"run": bool, "reason": str}} from the K1 summary; a missing summary or model means "do not run"."""
    out = {}
    summ = None
    if k1_summary_path and Path(k1_summary_path).exists():
        summ = json.loads(Path(k1_summary_path).read_text(encoding="utf-8"))
    for m in models_if_fit:
        e = ((summ or {}).get("models") or {}).get(m)
        if summ is None:
            out[m] = {"run": False, "reason": f"no K1 summary at {k1_summary_path}"}
        elif e is None or e.get("missing"):
            out[m] = {"run": False, "reason": "model not measured by K1 (missing from the store)"}
        elif e.get("fits_8gb_at_default"):
            out[m] = {"run": True, "reason": f"K1: fully on GPU at default ctx {e.get('default_ctx')} "
                                             f"(size_vram {e.get('size_vram')} of {e.get('size')})"}
        else:
            out[m] = {"run": False, "reason": f"K1: not fully on GPU at default ctx {e.get('default_ctx')} "
                                              f"(size_vram {e.get('size_vram')} of {e.get('size')})"}
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
    plan = plan_for(args.mode, tiers, seeds, args.turns)
    if dry and args.mode == "validation" and (seeds or args.turns):
        plan = [(a, tuple(seeds or s), args.turns or t) for a, s, t in plan]
    models = [m for m in args.models.split(",") if m]
    fit = fit_decision([m for m in args.models_if_fit.split(",") if m], args.k1_summary)
    models += [m for m, d in fit.items() if d["run"]]
    mit_argv = client_trim = None
    if args.mode == "mitigation":
        mit_argv, client_trim = check_mitigation_args(args, models)
    exe = args.ollama_exe or bc.PINNED["ollama_exe"]
    versions = bc.versions_record(exe)
    problems = bc.version_problems(versions)
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
        if problems and not args.allow_version_mismatch:
            emit({"record": "refused", "reasons": problems})
            log(f"refused (versions): {problems}")
            return EXIT_ERR
        rules_in_use = agent.RULE_IDS
        refused = {}
        if args.mode in ("real", "mitigation") and args.rules_from:
            vrows = agent.read_rows(Path(args.rules_from))
            pf = agent.validation_preflight_per_model([(Path(args.rules_from).name, vrows)], CALL2, models)
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
                report = agent.evaluate_gates(rows, call2_tools=CALL2)
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
