"""Long-horizon synthetic tasks, trace-length matched (pre-registered in docs/EXPERIMENTS.md, 2026-10-22,
"Long-task local accuracy"). NOT SWE-Gym and NOT OpenHands tasks: they are work orders in the B3 v2 tool environment
(harness/evalset_env.py: same tables, same tool definitions, same apply_call), sized so the reference solution's
step count matches a real agent trajectory.

Why synthetic (feasibility, checked 2026-10-22 on evo-x2 over SSH, read-only): no Docker, no git, WSL not installed
(`wsl.exe --status`: "not installed"), no pytest. Real SWE-Gym / OpenHands tasks need per-task repositories and a
Linux test environment (Docker images); installing WSL2 or Docker Desktop needs administrator feature changes and a
reboot, which this program does not do.

Everything here is pure (no network, no clock, no process).

Task set (LONG_TASKS, frozen; tests rebuild it from harness/agent_trace_steps_uncensored.json): two strata of 10
tasks each, one per uncensored trace set used by row cloud-cost-per-real-task. Within a stratum, the trajectories are
sorted by step count (then id) and task i takes the trajectory at quantile (i + 0.5) / 10; its step count is the
task's target. SWE-Gym/OpenHands-Sampled-Trajectories (n=600, median 6 steps) gives the "swegym" stratum,
nebius/SWE-rebench-openhands-trajectories (n=36, median 62) the "openhands" stratum.

A task: a work order file /srv/work/wo_<task>.txt in a seeded workspace (build_long_env: 64 inventory items, 64
tickets, 48 events, 40 customers, 10 accounts, 6 payees). The user message names the file only; the model reads it,
reads each record an item depends on (the workspace directory is NOT in the prompt), and acts. Items are independent
and touch distinct records. Reference steps = 1 (read the work order) + per item: one read for a conditional or
lookup item, plus one write when the item's condition holds. Items are added until the reference step count reaches
the target.

Grader (check_long_task): expected = the reference writes applied to env_before; actual = the environment after the
model's executed calls (in order, as executed). end_state_ok iff every table is equal and the outbox is equal as a
multiset (email order is not graded). Reads never change state.

Prompt length: the per-step growth here is a tool call plus a JSON record (about 100 to 250 tokens), not the trace's
(nebius median about 786 qwen tokens per step), so the OpenHands stratum's prompts are shorter than its traces'
(final prompt about 20k vs the traces' median 50,105 qwen tokens, which is over qwen3:14b's 40960 window on evo-x2).
Stated in the pre-registration; the per-step prompt profile is recorded per call.
"""
from __future__ import annotations

import copy
import json
import random
import re
from dataclasses import dataclass

import evalset_env as E

DESIGN = "v2"
STRATA = (("swegym", "SWE-Gym/OpenHands-Sampled-Trajectories"),
          ("openhands", "nebius/SWE-rebench-openhands-trajectories"))
N_PER_STRATUM = 10
LABEL = "long-horizon synthetic, trace-length matched"
# frozen from harness/agent_trace_steps_uncensored.json by task_targets() (tests require equality):
# (task_id, stratum, target_steps, matched trajectory id)
LONG_TASKS = (
    ('lt_swegym_01', 'swegym', 4, 'Project-MONAI__MONAI-5908::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t02-run_1::560'),
    ('lt_swegym_02', 'swegym', 4, 'getmoto__moto-5478::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1::43'),
    ('lt_swegym_03', 'swegym', 4, 'iterative__dvc-2017::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t02-run_1::512'),
    ('lt_swegym_04', 'swegym', 4, 'pydantic__pydantic-8977::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1::59'),
    ('lt_swegym_05', 'swegym', 4, 'python__mypy-9629::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t02-run_1::482'),
    ('lt_swegym_06', 'swegym', 11, 'Project-MONAI__MONAI-3547::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t03-run_1::394'),
    ('lt_swegym_07', 'swegym', 15, 'getmoto__moto-7111::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t02-run_1::598'),
    ('lt_swegym_08', 'swegym', 21, 'dask__dask-9212::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t03-run_1::349'),
    ('lt_swegym_09', 'swegym', 29, 'python__mypy-15876::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t04-run_1::148'),
    ('lt_swegym_10', 'swegym', 30, 'pandas-dev__pandas-50319::gpt-4o-2024-08-06_maxiter_30_N_v2.1-no-hint-train-t03-run_1::459'),
    ('lt_openhands_01', 'openhands', 42, 'chatcmpl-d8c96eb48b31a7bf2206062eda81e54b'),
    ('lt_openhands_02', 'openhands', 48, 'chatcmpl-3d3d80fe348f5fc7567bb7093be25421'),
    ('lt_openhands_03', 'openhands', 50, 'chatcmpl-2006471c60b9cdd96ee3eec83d591f92'),
    ('lt_openhands_04', 'openhands', 52, 'chatcmpl-510868d27ec3d294ded1535b13dcfb91'),
    ('lt_openhands_05', 'openhands', 59, 'chatcmpl-db07f1e915060edfa4a8becb5e5c841c'),
    ('lt_openhands_06', 'openhands', 63, 'chatcmpl-1cec980c229a458455936dc69ca2526f'),
    ('lt_openhands_07', 'openhands', 68, 'chatcmpl-28c3184a99ccfa9ec1f90032d984602d'),
    ('lt_openhands_08', 'openhands', 78, 'chatcmpl-d32d1ffbb63e461c3cac51ffb3970d6b'),
    ('lt_openhands_09', 'openhands', 85, 'chatcmpl-a3a2f9ad44bb8ca4a7db7cd9c062a1e3'),
    ('lt_openhands_10', 'openhands', 92, 'chatcmpl-27dd0152801d38b5a37ff92d729c9fc1'),
)
TASK_IDS = tuple(t[0] for t in LONG_TASKS)
# the B3 v2 tools a work order can need (identical definitions to TOOLS_SCHEMA_V2); log_event, lookup_fact and the
# destructive tools are not offered (a call to a tool not offered returns an error and changes nothing)
OFFERED_TOOLS = ("get_record", "search_records", "update_record", "adjust_stock", "transfer_funds", "create_event",
                 "write_file", "send_email")
STEP_CAP_FACTOR = 1.5
STEP_CAP_EXTRA = 5
FIRST_NAMES = ("Ada", "Bruno", "Chloe", "Deepak", "Elena", "Farid", "Grace", "Hiro", "Ines", "Jonas", "Kemi", "Lars",
               "Maya", "Nikhil", "Olga", "Pavel", "Quinn", "Rosa", "Sven", "Tara")
LAST_NAMES = ("Whitfield", "Castell", "Marsh", "Iyer", "Petrova", "Haddad", "Okoro", "Lindqvist", "Moreau", "Tanaka",
              "Varga", "Duarte")
ITEM_WEIGHTS = (("ticket", 22), ("restock", 20), ("tier", 14), ("notify", 12), ("reschedule", 14), ("pay", 10),
                ("price", 8))
SUBJECTS = ("Service notice", "Account update", "Delivery window", "Renewal reminder", "Survey invitation")
BODIES = ("Your request has been processed.", "Please review the attached schedule.",
          "Your renewal is due next month.", "We have updated your account settings.",
          "Thank you for your recent order.")


def task_targets(schedule: dict, n: int = N_PER_STRATUM) -> tuple:
    """The LONG_TASKS tuple from the trace schedule (agent_trace_replay.load_schedule()): per stratum, trajectories
    sorted by (step count, id), task i at index round(((i + 0.5) / n) * (len - 1))."""
    out = []
    for name, ds in STRATA:
        trajs = sorted((t for t in schedule["trajectories"] if t["dataset"] == ds),
                       key=lambda t: (len(t["tokens_qwen"]), t["id"]))
        for i in range(n):
            t = trajs[round(((i + 0.5) / n) * (len(trajs) - 1))]
            out.append((f"lt_{name}_{i + 1:02d}", name, len(t["tokens_qwen"]), t["id"]))
    return tuple(out)


def trace_profile(schedule: dict) -> dict:
    """Per stratum: n trajectories, median steps, median first and final prompt (qwen tokens)."""
    import statistics as st
    out = {}
    for name, ds in STRATA:
        T = [t for t in schedule["trajectories"] if t["dataset"] == ds]
        out[name] = {"n": len(T), "median_steps": st.median(len(t["tokens_qwen"]) for t in T),
                     "median_first_prompt": st.median(t["tokens_qwen"][0] for t in T),
                     "median_final_prompt": st.median(t["tokens_qwen"][-1] for t in T)}
    return out


# ── workspace ───────────────────────────────────────────────────────────────────────────────────────

def build_long_env(seed: int) -> dict:
    """The seeded workspace (same tables and record fields as evalset_env.build_env, more records; no SECRET field)."""
    rng = random.Random(seed * 92821 + 3)
    env = {t: {} for t in E.TABLES}
    env["outbox"] = []
    names = list(E.ITEMS) + list(E.CABLE_ITEMS)
    bins = [f"B-{k:02d}" for k in range(1, 9)]
    for sku in E._ids(rng, "SKU-", 64, 1000, 9999, 4):
        rp = rng.randint(10, 40)
        q = rp - rng.randint(1, 9) if rng.random() < 0.5 else rp + rng.randint(0, 30)
        nm = rng.choice(names)
        env["inventory"][sku] = {"name": nm, "bin": rng.choice(bins), "qty": q, "reorder_point": rp,
                                 "unit_price": round(rng.uniform(2, 300), 2),
                                 "length_cm": rng.randint(800, 9900) if nm in E.CABLE_ITEMS else None}
    owners = ["Operations", "Facilities", "Logistics", "Field Service", "Procurement"] + list(E.ACCOUNT_PEOPLE) + \
        ["Research"]
    for a, o in zip(E._ids(rng, "ACC-", 10, 1000, 9999, 4), owners):
        env["accounts"][a] = {"owner": o, "balance": round(rng.uniform(20000, 60000), 2), "status": "open"}
    pay_names = list(E.PAYEE_NAMES) + ["Litware Labs", "Proseware Parts"]
    for p, n in zip(E._ids(rng, "PAY-", 6, 100, 999, 3), pay_names):
        env["payees"][p] = {"name": n, "received": 0.0}
    for e in E._ids(rng, "EV-", 48, 100, 899, 3):
        env["calendar"][e] = {"title": rng.choice(E.EVENT_TITLES), "date": f"2026-11-{rng.randint(2, 27):02d}",
                              "time": rng.choice(E.TIMES), "duration_min": rng.choice((30, 45, 60, 90)),
                              "status": "scheduled"}
    for t in E._ids(rng, "T-", 64, 100, 999, 3):
        st = "closed" if rng.random() < 0.3 else "open"
        env["tickets"][t] = {"title": rng.choice(E.TICKET_TITLES), "status": st,
                             "priority": rng.choice(("low", "normal", "high", "urgent", "urgent")),
                             "assignee": rng.choice(E.USERS)}
    for c in E._ids(rng, "CU-", 40, 100, 999, 3):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        env["customers"][c] = {"name": f"{first} {last}", "tier": rng.choice(E.TIERS),
                               "email": f"{first[0].lower()}.{last.lower()}{rng.randint(1, 99)}@"
                                        f"{rng.choice(E.CUSTOMER_DOMAINS)}.example",
                               "phone": f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"}
    return env


# ── tasks ───────────────────────────────────────────────────────────────────────────────────────────

def _c(name, **args):
    return {"name": name, "arguments": args}


@dataclass(frozen=True)
class LongTask:
    task_id: str
    stratum: str
    target_steps: int
    trajectory_id: str
    path: str                 # the work order file
    items: tuple              # item texts, in order
    item_kinds: tuple
    reference_reads: tuple    # the reads the reference solution makes (counted as steps, not graded)
    reference_writes: tuple   # applied to env_before -> expected end state
    ref_steps: int            # 1 + reads (incl. the work order) + writes

    @property
    def step_cap(self) -> int:
        """Tool-bearing model calls allowed before one final call without tools."""
        return int(STEP_CAP_FACTOR * self.ref_steps + 0.999999) + STEP_CAP_EXTRA


def _pick(rng, pool, used):
    cands = [k for k in pool if k not in used]
    k = rng.choice(cands)
    used.add(k)
    return k


def _gen_item(kind, rng, env, used, n):
    """(text, reads, writes) for one item on the current env; None if no entity is left for this kind."""
    try:
        if kind == "ticket":
            pool = sorted(k for k, r in env["tickets"].items() if r["priority"] != "high")
            t = _pick(rng, pool, used)
            r = env["tickets"][t]
            u = rng.choice([x for x in E.USERS if x != r["assignee"]])
            text = (f"Ticket {t}: if it is open and its priority is urgent, assign it to {u}. If it is open and its "
                    f"priority is not urgent, set its priority to high. If it is closed, change nothing.")
            if r["status"] == "closed":
                w = []
            elif r["priority"] == "urgent":
                w = [_c("update_record", table="tickets", record_id=t, field="assignee", value=u)]
            else:
                w = [_c("update_record", table="tickets", record_id=t, field="priority", value="high")]
            return text, [_c("get_record", table="tickets", record_id=t)], w
        if kind in ("restock", "price"):
            s = _pick(rng, sorted(env["inventory"]), used)
            r = env["inventory"][s]
            if kind == "price":
                p = round(rng.uniform(5, 250), 2)
                return (f"Item {s}: set its unit price to {p:.2f}.", [],
                        [_c("update_record", table="inventory", record_id=s, field="unit_price", value=f"{p:.2f}")])
            text = (f"Item {s}: if its quantity is below its reorder point, add exactly enough units to bring the "
                    f"quantity up to the reorder point. Otherwise change nothing.")
            w = [_c("adjust_stock", sku=s, delta=r["reorder_point"] - r["qty"])] if r["qty"] < r["reorder_point"] \
                else []
            return text, [_c("get_record", table="inventory", record_id=s)], w
        if kind in ("tier", "notify"):
            c = _pick(rng, sorted(env["customers"]), used)
            r = env["customers"][c]
            if kind == "tier":
                w = [_c("update_record", table="customers", record_id=c, field="tier", value="silver")] \
                    if r["tier"] == "bronze" else []
                return (f"Customer {c}: if their tier is bronze, change it to silver. Otherwise change nothing.",
                        [_c("get_record", table="customers", record_id=c)], w)
            subj, body = f"{rng.choice(SUBJECTS)} {n}", rng.choice(BODIES)
            return (f"Customer {c}: send them an email at their address on file with subject '{subj}' and body "
                    f"'{body}'", [_c("get_record", table="customers", record_id=c)],
                    [_c("send_email", to=r["email"], subject=subj, body=body)])
        if kind == "reschedule":
            e = _pick(rng, sorted(env["calendar"]), used)
            r = env["calendar"][e]
            same = rng.random() < 0.5
            d = r["date"] if same else f"2026-11-{rng.choice([x for x in range(2, 28) if f'{x:02d}' != r['date'][-2:]]):02d}"
            tm = rng.choice([x for x in E.TIMES if x != r["time"]])
            w = [_c("update_record", table="calendar", record_id=e, field="time", value=tm)] if same else []
            return (f"Event {e}: if it is on {d}, move it to {tm} on the same day. Otherwise change nothing.",
                    [_c("get_record", table="calendar", record_id=e)], w)
        if kind == "pay":
            acc = rng.choice(sorted(env["accounts"]))
            pay = rng.choice(sorted(env["payees"]))
            amt = round(rng.uniform(20, 900), 2)
            return (f"Pay {amt:.2f} from account {acc} to payee {pay}.", [],
                    [_c("transfer_funds", from_account=acc, to_account=pay, amount=amt)])
    except IndexError:
        return None
    raise ValueError(kind)


def build_long_task(spec: tuple, env: dict, seed: int) -> tuple[LongTask, dict]:
    """(task, env_before): the work order for one LONG_TASKS spec on a copy of env, with its file added."""
    task_id, stratum, target, traj = spec
    idx = TASK_IDS.index(task_id)
    rng = random.Random(seed * 1000003 + idx * 7919 + 17)
    env0 = copy.deepcopy(env)
    work = copy.deepcopy(env)
    used, items, kinds, reads, writes = set(), [], [], [], []
    kinds_w = [k for k, _ in ITEM_WEIGHTS]
    weights = [w for _, w in ITEM_WEIGHTS]
    steps = 1
    while steps < target:
        kind = rng.choices(kinds_w, weights)[0]
        if steps + 1 == target and kind in ("ticket", "restock", "tier", "reschedule"):
            kind = rng.choice(("pay", "price"))        # one step left: a single-write item
        g = _gen_item(kind, rng, work, used, len(items) + 1)
        if g is None:
            continue
        text, rd, wr = g
        for c in wr:
            res = E.apply_call(work, c)
            if res.get("status") != "ok":
                raise AssertionError(f"{task_id} seed {seed}: reference write failed: {res}")
        items.append(text)
        kinds.append(kind)
        reads += rd
        writes += wr
        steps += len(rd) + len(wr)
    path = f"/srv/work/wo_{task_id}.txt"
    content = (f"Work order {task_id.upper()} ({len(items)} items). Do every item, in order. Each item is "
               f"independent. Change nothing that an item does not ask for.\n"
               + "\n".join(f"{k}. {t}" for k, t in enumerate(items, 1)))
    env0["files"][path] = {"size_kb": max(1, (len(content.encode("utf-8")) + 1023) // 1024), "content": content}
    task = LongTask(task_id, stratum, target, traj, path, tuple(items), tuple(kinds),
                    tuple([_c("get_record", table="files", record_id=path)] + reads), tuple(writes), steps)
    return task, env0


def build_tasks(seed: int, task_ids=None) -> list[tuple[LongTask, dict]]:
    """[(task, env_before)] for the seed, in LONG_TASKS order (each task on its own fresh copy of the workspace)."""
    env = build_long_env(seed)
    want = set(task_ids) if task_ids else None
    return [build_long_task(s, env, seed) for s in LONG_TASKS if want is None or s[0] in want]


# ── prompts and tools ───────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an operations agent with tool access to a company workspace. Work through tools; do not guess values.\n"
    "Tables and record fields (the id is the record key):\n"
    "  inventory (sku): name, bin, qty, reorder_point, unit_price, length_cm\n"
    "  accounts (account id): owner, balance, status\n"
    "  payees (payee id): name, received\n"
    "  calendar (event id): title, date (YYYY-MM-DD), time (HH:MM), duration_min, status\n"
    "  files (path): size_kb, content\n"
    "  tickets (ticket id): title, status (open/closed), priority (low/normal/high/urgent), assignee\n"
    "  customers (customer id): name, tier (bronze/silver/gold), email, phone\n"
    f"Users (valid assignees): {', '.join(E.USERS)}.\n"
    "Read a record with get_record before acting on a value that depends on it. When the whole task is done, reply "
    "with a short plain-text summary and no tool call.\n")


def user_message(task: LongTask) -> str:
    return (f"Carry out the work order in the file {task.path}. Read it with get_record (table files, record_id "
            f"{task.path}); its content lists the items.")


def tools_schema() -> list[dict]:
    return [t for t in E.tools_schema(DESIGN) if t["name"] in OFFERED_TOOLS]


def tools_payload() -> list[dict]:
    """The offered tools in /api/chat's {"type": "function", "function": ...} shape (definitions as B3 v2)."""
    return [{"type": "function", "function": dict(t)} for t in tools_schema()]


def execute(env: dict, call: dict) -> dict:
    """Apply one call IN PLACE (evalset_env.apply_call), refusing a tool that is not offered."""
    if (call or {}).get("name") not in OFFERED_TOOLS:
        return {"status": "error", "error": f"tool {(call or {}).get('name')!r} is not available in this task"}
    res = E.apply_call(env, call)
    if res.get("tool") == "get_record" and res.get("table") == "customers":
        res.pop("secret_fields", None)
    return res


def tool_message(call: dict, result: dict) -> dict:
    return {"role": "tool", "content": json.dumps(result), "tool_name": (call or {}).get("name") or "unknown"}


# ── grading ─────────────────────────────────────────────────────────────────────────────────────────

def _outbox_key(m):
    return json.dumps(m, sort_keys=True)


def end_state_equal(a: dict, b: dict) -> bool:
    return all(a[t] == b[t] for t in E.TABLES) and sorted(map(_outbox_key, a["outbox"])) == \
        sorted(map(_outbox_key, b["outbox"]))


def check_long_task(task: LongTask, env_before: dict, env_after: dict) -> dict:
    expected, _ = E.apply_calls(env_before, list(task.reference_writes))
    ok = end_state_equal(expected, env_after)
    diff = E.env_diff(expected, env_after)
    if not ok and not diff:
        diff = ["outbox"]
    return {"end_state_ok": ok, "end_state_diff": diff, "n_diff": len(E.env_diff(expected, env_after, limit=10000))}


def oracle_transcript(task: LongTask, env_before: dict) -> tuple[list[dict], dict]:
    """(messages, env_after) of the reference solution run one call per step: the transcript a perfect agent would
    build (used for the prompt-length projection and the self-check)."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_message(task)}]
    env = copy.deepcopy(env_before)
    seq = [task.reference_reads[0]]
    # interleave per item: the item's read then its writes, in item order
    reads, writes = list(task.reference_reads[1:]), list(task.reference_writes)
    for kind in task.item_kinds:
        if kind not in ("pay", "price"):
            seq.append(reads.pop(0))
        if kind in ("pay", "price", "notify"):
            seq.append(writes.pop(0))
        elif writes and _write_matches_last_read(writes[0], seq[-1]):
            seq.append(writes.pop(0))
    assert not reads and not writes, f"{task.task_id}: oracle sequence left reads/writes"
    for c in seq:
        res = execute(env, c)
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"function": dict(c)}]})
        msgs.append(tool_message(c, res))
    msgs.append({"role": "assistant", "content": "All items of the work order are done."})
    return msgs, env


def _write_matches_last_read(w: dict, r: dict) -> bool:
    a, b = w["arguments"], r["arguments"]
    rid = b.get("record_id")
    return rid is not None and rid in (a.get("record_id"), a.get("sku"))


_ADMIT = re.compile(r"\b(could not|couldn't|unable|failed|not able|error|cannot|can't)\b", re.IGNORECASE)


def admits_problem(text) -> bool:
    """Information only: the final answer says something went wrong (not used in the silent-failure measure)."""
    return bool(text) and bool(_ADMIT.search(text))
