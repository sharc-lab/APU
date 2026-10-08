"""Parse a harness's command line without running the harness.

probe(main_fn, argv) calls main_fn with argparse.ArgumentParser.parse_args patched so that the first parse returns
control to the caller (by raising an internal exception) right after argparse has accepted or rejected argv. Nothing
after the parse line in main_fn runs: no hostname check, no server start, no file written, no queue advance. Used by
scripts/t2s_week_queue.py's dry run and by harness/t2s_r2_agent.py's capability check (which flags and --mode
choices the deployed x2_r2_agent.py actually has).

Works for a main(argv) that calls ap.parse_args(argv) and for a main() that calls ap.parse_args() on sys.argv
(pass use_sys_argv=True; sys.argv is restored afterwards).
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from unittest import mock


class _Parsed(Exception):
    def __init__(self, parser, namespace):
        super().__init__("parsed")
        self.parser, self.namespace = parser, namespace


def parser_options(parser) -> dict:
    """{option_string: choices or None} for every optional argument of parser."""
    out = {}
    for act in parser._actions:  # noqa: SLF001 (argparse has no public accessor)
        for s in act.option_strings:
            out[s] = tuple(act.choices) if act.choices is not None else None
    return out


def probe(main_fn, argv, use_sys_argv=False, prog="probe"):
    """Returns {"ok": bool, "namespace": dict|None, "options": {flag: choices}, "error": str|None}. ok is False when
    argparse rejected argv (its usage message is in error) or main_fn returned/raised before parsing at all."""
    orig = argparse.ArgumentParser.parse_args

    def intercept(self, args=None, namespace=None):
        ns = orig(self, args, namespace)
        raise _Parsed(self, ns)

    err = io.StringIO()
    saved_argv = sys.argv
    try:
        if use_sys_argv:
            sys.argv = [prog, *argv]
        with mock.patch.object(argparse.ArgumentParser, "parse_args", intercept), \
                contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            if use_sys_argv:
                main_fn()
            else:
                main_fn(list(argv))
    except _Parsed as p:
        return {"ok": True, "namespace": vars(p.namespace), "options": parser_options(p.parser), "error": None}
    except SystemExit as e:
        return {"ok": False, "namespace": None, "options": {}, "error": (err.getvalue().strip() or f"exit {e.code}")}
    finally:
        sys.argv = saved_argv
    return {"ok": False, "namespace": None, "options": {}, "error": "main returned without parsing a command line"}
