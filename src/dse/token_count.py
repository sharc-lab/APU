"""Prompt-token counting for the router from the server or the model's tokenizer (operator decision 2026-11-13: "Use
the server's tokenize endpoint, or the model's tokenizer if the server can't count"). Replaces the chars/4 x
calibration-ratio estimate in src/dse/router.py wherever a counter is passed (Router(token_counter=...)).

Two counters, both returning (per-message counts, tools/framing tokens) for one request, the shape Router._count uses:

  LlamaServerCounter   llama-server: POST /apply-template (the model's own chat template, tools included) then POST
                       /tokenize on the rendered prompt. Exact for the prompt the server will evaluate. Per-message
                       counts are prefix differences of the rendered prompt, cached by prefix, adjusted so the sum is
                       exactly the whole render.
  TokenizerCounter     Ollama has no tokenize endpoint, so the model's tokenizer (a Hugging Face tokenizer.json,
                       `tokenizers` library) tokenizes each message as the family's chat template frames it. The
                       content tokens are exact; the framing follows the family template as written below. ChatML
                       (Qwen) equals the model's own Hugging Face chat template token for token, tools included
                       (tests/test_token_count.py renders the real template). NOT yet checked against Ollama's
                       prompt_eval_count, whose Go template can differ from the Hugging Face one (to run on evo-x2
                       after the BIOS splits). The llama3 framing is not verified and make_counter never selects it.

make_counter() picks the server when it can count and the tokenizer otherwise, and raises TokenCountUnavailable when
neither is available: the router never falls back to chars/4 silently.
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from typing import Any, Callable

GEN_PROMPT = {"chatml": "<|im_start|>assistant\n", "llama3": "<|start_header_id|>assistant<|end_header_id|>\n\n"}


class TokenCountUnavailable(RuntimeError):
    """Neither the server nor a model tokenizer can count this request."""


def _key(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _content(m: dict) -> str:
    c = m.get("content")
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict)) if isinstance(c, list) else str(c)


def _http_post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class LlamaServerCounter:
    source = "llama-server /apply-template + /tokenize"

    def __init__(self, base_url: str, timeout: float = 10.0, post: Callable[[str, dict, float], dict] = _http_post):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._post = post
        self._cache: dict[str, int] = {}

    def n_prompt(self, messages: list[dict], tools: list | None = None) -> int:
        k = _key([messages, tools])
        if k not in self._cache:
            body: dict[str, Any] = {"messages": messages}
            if tools:
                body["tools"] = tools
            prompt = self._post(f"{self.base_url}/apply-template", body, self.timeout)["prompt"]
            toks = self._post(f"{self.base_url}/tokenize", {"content": prompt, "add_special": False},
                              self.timeout)["tokens"]
            self._cache[k] = len(toks)
        return self._cache[k]

    def available(self) -> bool:
        try:
            self._post(f"{self.base_url}/tokenize", {"content": "ok", "add_special": False}, self.timeout)
            return True
        except Exception:
            return False

    def counts(self, messages: list[dict], tools: list | None = None) -> tuple[list[int], int]:
        total = self.n_prompt(messages, tools)
        prefix = [self.n_prompt(messages[:i + 1]) for i in range(len(messages))]
        counts = [max(0, b - a) for a, b in zip([0] + prefix[:-1], prefix)]
        framing = max(0, total - sum(counts))
        if sum(counts) > total:            # a template that renders history shorter than the prefix (thinking strip)
            counts[-1] = max(0, counts[-1] - (sum(counts) - total))
            framing = total - sum(counts)
        return counts, framing


class TokenizerCounter:
    source = "model tokenizer (tokenizer.json) + family chat template"

    def __init__(self, family: str, encode: Callable[[str], int] | None = None, tokenizer_json: str | None = None):
        if family not in GEN_PROMPT:
            raise ValueError(f"unknown template family {family!r}; known: {sorted(GEN_PROMPT)}")
        self.family = family
        if encode is None:
            if tokenizer_json is None:
                raise TokenCountUnavailable("TokenizerCounter needs encode= or tokenizer_json=")
            from tokenizers import Tokenizer   # deferred: only this counter needs it
            tok = Tokenizer.from_file(str(tokenizer_json))
            encode = lambda s: len(tok.encode(s, add_special_tokens=False).ids)   # noqa: E731
        self._encode = encode

    # ── the family templates (as written by the model's chat template, tools in the system block)
    def _tools_block(self, tools: list | None) -> str:
        if not tools:
            return ""
        lines = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools)
        if self.family == "chatml":
            return ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided "
                    "with function signatures within <tools></tools> XML tags:\n<tools>\n" + lines + "\n</tools>\n\n"
                    "For each function call, return a json object with function name and arguments within "
                    "<tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": "
                    "<args-json-object>}\n</tool_call>")
        return ("Given the following functions, please respond with a JSON for a function call with its proper "
                "arguments that best answers the given prompt.\n\n" + lines)

    def _assistant_text(self, m: dict) -> str:
        text = _content(m)
        for tc in m.get("tool_calls") or []:
            f = tc.get("function", tc)
            args = f.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
            call = json.dumps({"name": f.get("name"), "arguments": args}, ensure_ascii=False)
            text += ("\n<tool_call>\n" + call + "\n</tool_call>") if self.family == "chatml" else call
        return text

    def _frame(self, role: str, text: str) -> str:
        if self.family == "chatml":
            return f"<|im_start|>{role}\n{text}<|im_end|>\n"
        return f"<|start_header_id|>{role}<|end_header_id|>\n\n{text}<|eot_id|>"

    def _render_one(self, m: dict) -> str:
        role = m.get("role")
        if role == "assistant":
            return self._frame("assistant", self._assistant_text(m))
        if role == "tool":
            if self.family == "chatml":
                return self._frame("user", "<tool_response>\n" + _content(m) + "\n</tool_response>")
            return self._frame("ipython", _content(m))
        return self._frame(role, _content(m))

    def counts(self, messages: list[dict], tools: list | None = None) -> tuple[list[int], int]:
        counts = []
        sys_tools = self._tools_block(tools)
        for i, m in enumerate(messages):
            if i == 0 and m.get("role") == "system" and sys_tools:
                # the tools block shares the system turn; it is counted once in framing below
                counts.append(self._encode(self._frame("system", _content(m) + "\n\n")))
            else:
                counts.append(self._encode(self._render_one(m)))
        framing = self._encode(GEN_PROMPT[self.family])
        if self.family == "llama3":
            framing += self._encode("<|begin_of_text|>")
        if sys_tools:
            framing += self._encode(sys_tools)
            if not (messages and messages[0].get("role") == "system"):
                framing += self._encode(self._frame("system", ""))
        return counts, framing


def family_of(model: str) -> str | None:
    """The template family make_counter may use for a model's tokenizer count. Only ChatML (Qwen) is selected: it
    matches the model's own chat template exactly (tests/test_token_count.py). The llama3 framing above does NOT match
    (2026-11-13 check: +16% without tools, +85% with tools against the HF Llama 3.1 template, and Ollama renders
    Llama with its own template), so a Llama model on Ollama gets TokenCountUnavailable until that is fixed and
    checked against server counts."""
    return "chatml" if model.lower().startswith("qwen") else None


def make_counter(runtime_kind: str, model: str, *, base_url: str | None = None, tokenizer_json: str | None = None,
                 server_counter: LlamaServerCounter | None = None):
    """The server's count when it can count (llama-server), else the model's tokenizer; never chars/4."""
    if runtime_kind == "llama_server" or runtime_kind == "llama-server":
        srv = server_counter or (LlamaServerCounter(base_url) if base_url else None)
        if srv is not None and srv.available():
            return srv
    fam = family_of(model)
    if fam is not None and tokenizer_json:
        return TokenizerCounter(fam, tokenizer_json=tokenizer_json)
    raise TokenCountUnavailable(
        f"no token count for {model} on {runtime_kind}: the server cannot count"
        f"{'' if base_url else ' (no server URL)'} and no tokenizer.json was given"
        f"{'' if fam else ' (unknown template family)'}")
