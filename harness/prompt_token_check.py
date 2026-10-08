"""Validity check for the R2 mechanism job's render-only prompt (harness/x2_r2_mechanism.py).

The mechanism job reads Ollama's "_debug_render_only" reply as the prompt the model actually saw after truncation.
This module tests that claim per checked call by comparing two token counts:
  (a) render_tokens: the render-only text tokenized with the model's own tokenizer, llama.cpp's llama-tokenize
      (TOKENIZE_EXE) against the model's GGUF blob, resolved from the Ollama manifest
      (manifests/<registry>/<namespace>/<name>/<tag> -> layer application/vnd.ollama.image.model -> blobs/sha256-<hex>).
      Command: TOKENIZE_EXE -m <blob> --stdin --ids --show-count --no-escape --log-disable, stdin = a temp file holding
      the UTF-8 text (never a command-line argument: prompts are tens of thousands of tokens). --stdin rather than -f:
      the common -f handler drops one trailing newline. --no-escape: the common parser would otherwise turn a literal
      backslash-n in tool JSON into a newline. Special tokens are parsed and BOS is added per the GGUF's
      add_bos_token, which is what llama-server does with the prompt Ollama sends it (the llama3.1 Ollama template
      has no <|begin_of_text|> of its own; a doubled BOS at the start of the ids is recorded as double_bos).
  (b) prompt_eval_count from a fresh, uncached /api/chat request with exactly the same messages and options and
      num_predict 1, sent after the model was unloaded (keep_alive 0, then /api/ps polled until it is gone), so no
      KV or prompt cache can shorten the count.
match = |a - b| <= 1% of b. A tier is mechanism_citable only when it has at least MIN_CHECKS checks and every one of
them matched (an errored check counts as not matched; nothing is skipped).

Assumed llama-tokenize output (llama.cpp tools/tokenize/tokenize.cpp; --help text captured on evo-x2 2026-10-07 lists
--ids "only print the token IDs, in a Python-parseable list form like [1, 2, ...]" and --show-count "print the total
number of tokens"): one line "[128000, 128006, 9125]" and one line "Total number of tokens: 3". parse_tokenize_output
takes the count from the ids list and, when both are present, requires them to agree.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

TOKENIZE_EXE = r"C:\apu\bin\llama-b10970\llama-tokenize.exe"
TOKENIZE_TIMEOUT_S = 600
MATCH_REL_TOL = 0.01
MIN_CHECKS = 5
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
DEFAULT_REGISTRY = "registry.ollama.ai"
NOT_CITABLE = "render does not show what the model saw; mechanism result not citable"


# ── Ollama blob resolution ──────────────────────────────────────────────────────────────────────────

def manifest_path(models_dir, model: str) -> Path:
    """Ollama model reference -> manifest file. "llama3.1:8b" -> manifests/registry.ollama.ai/library/llama3.1/8b;
    a tagless name gets "latest"; "ns/name:tag" and "host/ns/name:tag" keep their namespace / registry."""
    name, tag = model, "latest"
    last = model.rsplit("/", 1)[-1]
    if ":" in last:
        name, tag = model.rsplit(":", 1)
    parts = name.split("/")
    if len(parts) == 1:
        parts = [DEFAULT_REGISTRY, "library"] + parts
    elif len(parts) == 2:
        parts = [DEFAULT_REGISTRY] + parts
    return Path(models_dir) / "manifests" / Path(*parts) / tag


def resolve_model_blob(models_dir, model: str) -> Path:
    """The GGUF blob of `model` in an Ollama store; raises (FileNotFoundError / ValueError) rather than guessing."""
    mp = manifest_path(models_dir, model)
    if not mp.is_file():
        raise FileNotFoundError(f"no Ollama manifest for {model!r} at {mp}")
    man = json.loads(mp.read_text(encoding="utf-8"))
    digests = [l.get("digest") for l in man.get("layers") or [] if l.get("mediaType") == MODEL_MEDIA_TYPE]
    if len(digests) != 1 or not digests[0] or ":" not in digests[0]:
        raise ValueError(f"manifest {mp} has {len(digests)} model layers (need exactly one): {digests}")
    algo, hexd = digests[0].split(":", 1)
    blob = Path(models_dir) / "blobs" / f"{algo}-{hexd}"
    if not blob.is_file():
        raise FileNotFoundError(f"model blob {blob} for {model!r} not found")
    return blob


def default_models_dir() -> str:
    """OLLAMA_MODELS, else host_config's per-host store (under SYSTEM the profile default would be empty)."""
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    import host_config as hc
    d = hc._this_host_entry().get("ollama_models")
    if not d:
        raise RuntimeError("no OLLAMA_MODELS and this host has no ollama_models entry in host_config")
    return d


# ── llama-tokenize ──────────────────────────────────────────────────────────────────────────────────

_IDS = re.compile(r"\[\s*(-?\d+(?:\s*,\s*-?\d+)*)?\s*,?\s*\]")
_COUNT = re.compile(r"Total number of tokens:\s*(\d+)")


def parse_tokenize_output(stdout: str) -> dict:
    """{"n_tokens", "ids_head", "n_ids", "n_shown"} from llama-tokenize --ids --show-count stdout; raises ValueError
    when neither is found or the two disagree."""
    ids = None
    for m in _IDS.finditer(stdout or ""):
        body = m.group(1)
        cand = [int(x) for x in body.split(",")] if body else []
        if ids is None or len(cand) > len(ids):
            ids = cand
    cm = _COUNT.search(stdout or "")
    shown = int(cm.group(1)) if cm else None
    if ids is None and shown is None:
        raise ValueError(f"llama-tokenize output has neither an id list nor a token count: {stdout[:300]!r}")
    if ids is not None and shown is not None and len(ids) != shown:
        raise ValueError(f"llama-tokenize id list has {len(ids)} ids but reports {shown} tokens")
    n = len(ids) if ids is not None else shown
    return {"n_tokens": n, "n_ids": None if ids is None else len(ids), "n_shown": shown,
            "ids_head": (ids or [])[:4],
            "double_bos": bool(ids) and len(ids) >= 2 and ids[0] == ids[1]}


def tokenize_cmd(gguf, exe=TOKENIZE_EXE) -> list[str]:
    return [str(exe), "-m", str(gguf), "--stdin", "--ids", "--show-count", "--no-escape", "--log-disable"]


def count_tokens(text: str, gguf, exe=TOKENIZE_EXE, run=None, tmp_dir=None) -> dict:
    """Token count of `text` with the model's own tokenizer. The text goes through a temp file (stdin), written as
    UTF-8 bytes with no newline translation. Raises on any failure (non-zero exit, unparseable output)."""
    if run is None:
        from proc_util import run_hidden as run
    fd, tmp = tempfile.mkstemp(prefix="r2m_render_", suffix=".txt", dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        cmd = tokenize_cmd(gguf, exe)
        with open(tmp, "rb") as fin:
            p = run(cmd, stdin=fin, capture_output=True, timeout=TOKENIZE_TIMEOUT_S)
        out = p.stdout.decode("utf-8", "replace") if isinstance(p.stdout, bytes) else (p.stdout or "")
        err = p.stderr.decode("utf-8", "replace") if isinstance(p.stderr, bytes) else (p.stderr or "")
        if p.returncode != 0:
            raise RuntimeError(f"llama-tokenize exit {p.returncode}: {err[-300:]!r}")
        res = parse_tokenize_output(out)
        res["cmd"] = cmd
        res["text_bytes"] = len(text.encode("utf-8"))
        return res
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


# ── comparison and summaries (pure) ─────────────────────────────────────────────────────────────────

def compare(render_tokens, prompt_eval_count, tol=MATCH_REL_TOL) -> dict:
    if not isinstance(render_tokens, int) or not isinstance(prompt_eval_count, int) or prompt_eval_count <= 0:
        return {"diff": None, "rel_diff": None, "match": False}
    d = render_tokens - prompt_eval_count
    return {"diff": d, "rel_diff": d / prompt_eval_count, "match": abs(d) <= tol * prompt_eval_count}


def tier_summary(check_rows: list[dict], min_checks=MIN_CHECKS) -> dict:
    """{tier (arm_id): {n_checks, n_match, match_rate, n_post_overflow, n_pre_overflow, n_errors, max_abs_rel_diff,
    mechanism_citable, statement}} over r2m_prompt_check rows (latest per (tier, seed, turn, call))."""
    latest = {}
    for r in check_rows:
        latest[(r.get("arm_id"), r.get("seed"), r.get("turn_idx"), r.get("call_idx"))] = r
    by = {}
    for r in latest.values():
        by.setdefault(r.get("arm_id"), []).append(r)
    out = {}
    for tier, rs in sorted(by.items(), key=lambda kv: str(kv[0])):
        n = len(rs)
        n_match = sum(1 for r in rs if r.get("match") is True)
        rels = [abs(r["rel_diff"]) for r in rs if isinstance(r.get("rel_diff"), (int, float))]
        citable = n >= min_checks and n_match == n
        out[tier] = {"n_checks": n, "n_match": n_match, "match_rate": n_match / n if n else None,
                     "n_post_overflow": sum(1 for r in rs if r.get("post_overflow")),
                     "n_pre_overflow": sum(1 for r in rs if not r.get("post_overflow")),
                     "n_errors": sum(1 for r in rs if r.get("error")),
                     "max_abs_rel_diff": max(rels) if rels else None,
                     "mechanism_citable": citable,
                     "statement": None if citable else (
                         NOT_CITABLE + (f" (only {n} checks, need {min_checks})" if n < min_checks else ""))}
    return out


class PromptChecker:
    """Fresh-load prompt_eval_count vs llama-tokenize count of the render, for one call. Needs a runtime with
    unload(model) (keep_alive 0 + /api/ps poll) and _post(path, body); body_fn builds the exact /api/chat body the
    real call sends (x2_r2_agent.native_chat_body). Every failure becomes an error in the returned fields."""

    def __init__(self, runtime, body_fn, models_dir=None, exe=TOKENIZE_EXE, run=None, tmp_dir=None):
        self.rt, self.body_fn, self.models_dir, self.exe = runtime, body_fn, models_dir, exe
        self.run, self.tmp_dir = run, tmp_dir
        self._blobs = {}

    def blob(self, model):
        if model not in self._blobs:
            self._blobs[model] = resolve_model_blob(self.models_dir or default_models_dir(), model)
        return self._blobs[model]

    def check(self, model, messages, num_ctx, tools, think, extra, rendered) -> dict:
        out = {"render_tokens": None, "check_prompt_eval_count": None, "error": None}
        errs = []
        if rendered is None:
            errs.append("render-only request returned no prompt")
        else:
            try:
                gguf = self.blob(model)
                tk = count_tokens(rendered, gguf, self.exe, self.run, self.tmp_dir)
                out.update({"render_tokens": tk["n_tokens"], "tokenize_cmd": tk["cmd"], "gguf": str(gguf),
                            "render_ids_head": tk["ids_head"], "double_bos": tk["double_bos"],
                            "render_bytes": tk["text_bytes"]})
            except Exception as e:
                errs.append(f"tokenize: {e!r}"[:400])
        try:
            unloaded = self.rt.unload(model)
        except Exception as e:
            unloaded = False
            errs.append(f"unload: {e!r}"[:300])
        out["unloaded_before_check"] = bool(unloaded)
        if not unloaded:
            errs.append("model still listed in /api/ps after unload; check request not sent (would be cached)")
        else:
            body = self.body_fn(model, [dict(m) for m in messages], num_ctx, tools, think, extra)
            body.setdefault("options", {})["num_predict"] = 1
            status, data, err, dt = self.rt._post("/api/chat", body)
            out.update({"check_http_status": status, "check_duration_s": dt,
                        "check_load_duration_ns": (data or {}).get("load_duration"),
                        "check_options": body.get("options")})
            if data is None:
                errs.append(f"check request: {status} {err}"[:400])
            else:
                out["check_prompt_eval_count"] = data.get("prompt_eval_count")
                if not isinstance(out["check_prompt_eval_count"], int):
                    errs.append("check response has no prompt_eval_count")
        out.update(compare(out["render_tokens"], out["check_prompt_eval_count"]))
        if errs:
            out["error"] = "; ".join(errs)
            out["match"] = False
        return out
