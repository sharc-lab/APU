"""Where a model's chat template comes from, per runtime (2026-10-08).

Why this exists: the same qwen3-4b-2507 weights wrote their R2 tool calls as text inside the JSON answer when served
by Ollama from a bare GGUF (`ollama create` with only "FROM <gguf>", no TEMPLATE), while library-pulled qwen3 models
made native calls (results/x2_r2_validation_v2b.jsonl, register R2-validation-v2b-runinfo). An outcome-table score
measured through a bare-GGUF Ollama tag is therefore a statement about that install path, not about the model alone,
so every outcome row records its template source.

Values of chat_template_source:
  ollama_library                 : an Ollama tag whose manifest carries a template layer (application/vnd.ollama.image
                                   .template), i.e. a registry-pulled library model (llama3.1:8b, qwen3:8b/14b/32b).
                                   chat_template_sha256 = that layer's digest (= sha256 of the template text).
  bare_gguf_ollama_create        : an Ollama tag created locally from a GGUF with no TEMPLATE (manifest has a model layer
                                   and no template layer; /api/show then reports the GGUF's own jinja chat template and
                                   the Modelfile shows "TEMPLATE {{ .Prompt }}"). chat_template_sha256 = sha256 of the
                                   GGUF's tokenizer.chat_template (read from the model blob).
  ollama_create_library_template : a tag this repo creates locally from a GGUF WITH the library Qwen3 TEMPLATE copied
                                   verbatim (CREATED_WITH_LIBRARY_TEMPLATE, e.g. qwen3-4b-2507-tools).
  gguf_embedded_llama_server     : llama-server started on a GGUF; it renders the GGUF's own tokenizer.chat_template
                                   (llama-server log: "llama_model_loader: - kv NN: tokenizer.chat_template", "chat
                                   template, example_format", "chat format: peg-native"). sha256 of that template.

Pure functions (classify_manifest, source_for_row) are unit-tested; the I/O helpers are cached per process and never
raise (a failure yields None fields, so recording the source can never break a measurement row).

CLI (read-only: manifests, GGUF metadata, GET /api/tags, POST /api/show; nothing is loaded or created):
  py -3.12 chat_template_source.py --probe            # one JSON line per Ollama tag and per GGUF, to stdout
"""
from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from pathlib import Path

_here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
for _p in (_here, _here.parent / "harness", Path(r"C:\apu\ovn")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SOURCES = ("ollama_library", "bare_gguf_ollama_create", "ollama_create_library_template",
           "gguf_embedded_llama_server")
BARE = "bare_gguf_ollama_create"
LLAMA_SERVER = "gguf_embedded_llama_server"
CREATED_WITH_LIBRARY_TEMPLATE = {"qwen3-4b-2507-tools"}
TEMPLATE_MEDIA_TYPE = "application/vnd.ollama.image.template"
MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"
DEFAULT_REGISTRY = "registry.ollama.ai"


def norm_tag(tag: str) -> str:
    """"qwen3-4b-2507:latest" -> "qwen3-4b-2507"; a tag with an explicit version is unchanged."""
    return tag[:-len(":latest")] if tag and tag.endswith(":latest") else tag


def manifest_path(models_dir, model: str) -> Path:
    """Same layout as prompt_token_check.manifest_path (kept local so this module imports nothing else)."""
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


def _digest_hex(d):
    return d.split(":", 1)[1] if d and ":" in d else d


def classify_manifest(tag: str, manifest: dict) -> dict:
    """{chat_template_source, template_layer_sha256 (None if no template layer), model_blob_digest}."""
    layers = manifest.get("layers") or []
    tpl = [l.get("digest") for l in layers if l.get("mediaType") == TEMPLATE_MEDIA_TYPE]
    mdl = [l.get("digest") for l in layers if l.get("mediaType") == MODEL_MEDIA_TYPE]
    if tpl:
        src = "ollama_create_library_template" if norm_tag(tag) in CREATED_WITH_LIBRARY_TEMPLATE else "ollama_library"
    else:
        src = BARE
    return {"chat_template_source": src, "template_layer_sha256": _digest_hex(tpl[0]) if tpl else None,
            "model_blob_digest": mdl[0] if mdl else None}


@functools.lru_cache(maxsize=None)
def gguf_template_sha256(path: str):
    """sha256 of the GGUF's tokenizer.chat_template (None if absent or unreadable)."""
    try:
        import gguf_meta
        t = gguf_meta.read_gguf_meta(str(path)).get("tokenizer.chat_template")
        return hashlib.sha256(t.encode("utf-8")).hexdigest() if isinstance(t, str) and t else None
    except Exception:
        return None


def _models_dir(models_dir=None):
    if models_dir:
        return models_dir
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return env
    try:
        import host_config as hc
        return hc._this_host_entry().get("ollama_models")
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def ollama_tag_template(tag: str, models_dir=None) -> dict:
    """Template source of an Ollama tag from its manifest on disk (no server needed). Never raises."""
    out = {"chat_template_source": None, "chat_template_sha256": None, "chat_template_evidence": None}
    try:
        md = _models_dir(models_dir)
        mp = manifest_path(md, tag)
        info = classify_manifest(tag, json.loads(mp.read_text(encoding="utf-8")))
        out["chat_template_source"] = info["chat_template_source"]
        if info["template_layer_sha256"]:
            out["chat_template_sha256"] = info["template_layer_sha256"]
            out["chat_template_evidence"] = "manifest template layer"
        elif info["model_blob_digest"]:
            blob = Path(md) / "blobs" / ("sha256-" + _digest_hex(info["model_blob_digest"]))
            out["chat_template_sha256"] = gguf_template_sha256(str(blob))
            out["chat_template_evidence"] = "manifest has no template layer; sha256 of the GGUF tokenizer.chat_template"
    except Exception as e:
        out["chat_template_evidence"] = f"unavailable: {e!r}"[:200]
    return out


def llama_server_template(gguf_path: str) -> dict:
    return {"chat_template_source": LLAMA_SERVER, "chat_template_sha256": gguf_template_sha256(str(gguf_path)),
            "chat_template_evidence": "llama-server renders the GGUF tokenizer.chat_template"}


# ── rows written before the field existed (derived in analysis, never written back) ─────────────────

def facts_from_probe(rows: list[dict]) -> dict:
    """{("ollama", tag) | ("gguf", path): {chat_template_source, chat_template_sha256}} from --probe output rows."""
    facts = {}
    for r in rows:
        if r.get("record") == "ollama_tag_template":
            facts[("ollama", norm_tag(r["tag"]))] = {k: r.get(k) for k in ("chat_template_source", "chat_template_sha256")}
        elif r.get("record") == "gguf_template":
            facts[("gguf", r["gguf"])] = {"chat_template_source": LLAMA_SERVER,
                                          "chat_template_sha256": r.get("chat_template_sha256")}
    return facts


def source_for_row(row: dict, model_map: dict, facts: dict, legacy_tag: dict | None = None,
                   ollama_configs=("ollama_default",), llama_configs=("llama_server",)) -> dict:
    """chat_template_source / chat_template_sha256 for an outcome row: the row's own fields if it recorded them,
    else derived from (model_id, config) via model_map {model_id: (ollama_tag, gguf)} and probe facts."""
    if row.get("chat_template_source"):
        return {"chat_template_source": row["chat_template_source"],
                "chat_template_sha256": row.get("chat_template_sha256"), "derived": False}
    m, c = row.get("model_id"), row.get("config")
    if m not in model_map:
        return {"chat_template_source": None, "chat_template_sha256": None, "derived": True}
    tag, gguf = model_map[m]
    if c in ollama_configs:
        tag = row.get("ollama_tag") or (legacy_tag or {}).get(m, tag)
        f = facts.get(("ollama", norm_tag(tag))) or {}
    elif c in llama_configs:
        f = facts.get(("gguf", gguf)) or {"chat_template_source": LLAMA_SERVER, "chat_template_sha256": None}
    else:
        f = {}
    return {"chat_template_source": f.get("chat_template_source"),
            "chat_template_sha256": f.get("chat_template_sha256"), "derived": True}


# ── probe (read-only) ───────────────────────────────────────────────────────────────────────────────

def _show(tag, base="http://127.0.0.1:11434"):
    import urllib.request
    req = urllib.request.Request(f"{base}/api/show", data=json.dumps({"model": tag}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def probe(tags, ggufs, models_dir=None, show_fn=_show) -> list[dict]:
    import socket
    from datetime import datetime, timezone
    host, ts = socket.gethostname(), datetime.now(timezone.utc).isoformat()
    out = []
    for t in tags:
        row = {"record": "ollama_tag_template", "host": host, "ts_utc": ts, "tag": t}
        row.update(ollama_tag_template(t, models_dir))
        try:
            s = show_fn(t)
            tpl = s.get("template") or ""
            row["api_show_template_sha256"] = hashlib.sha256(tpl.encode("utf-8")).hexdigest() if tpl else None
            row["api_show_template_len"] = len(tpl)
            row["api_show_modelfile_template_line"] = next(
                (l for l in (s.get("modelfile") or "").splitlines() if l.startswith("TEMPLATE")), None)
            row["api_show_capabilities"] = s.get("capabilities")
            row["api_show_parent_model"] = (s.get("details") or {}).get("parent_model")
            row["api_show_stop"] = [l.split(None, 1)[1].strip().strip('"') for l in
                                    (s.get("parameters") or "").splitlines() if l.split(None, 1)[:1] == ["stop"]]
        except Exception as e:
            row["api_show_error"] = repr(e)[:200]
        out.append(row)
    for g in ggufs:
        out.append({"record": "gguf_template", "host": host, "ts_utc": ts, "gguf": g,
                    "chat_template_sha256": gguf_template_sha256(g)})
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", action="store_true", required=True)
    ap.add_argument("--extra-tags", default="qwen3:30b-a3b,qwen3-4b-2507-tools")
    args = ap.parse_args(argv)
    import x2_outcome_table as x2ot
    tags = [t for t, _ in x2ot.MODEL_MAP.values()] + [t for t in args.extra_tags.split(",") if t]
    ggufs = [g for _, g in x2ot.MODEL_MAP.values()]
    for row in probe(list(dict.fromkeys(tags)), list(dict.fromkeys(ggufs))):
        print(json.dumps(row))


if __name__ == "__main__":
    main()
