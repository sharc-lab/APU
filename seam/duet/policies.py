"""DUET controls, rungs and ablations on synthetic shapes (reference rules of the DUET addendum, Step 4).

One engine; each named policy is a set of switches. P is the device prompt including this turn's
delta; Pc is the cloud transcript, which grows by dnew on every turn, local or cloud.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from seam.duet import params as prm
from seam.duet.device import cold_mem_ok, session_cap
from seam.duet.ledger import Ledger
from seam.duet.shapes import DTurn
from seam.duet.ttft import cold, ext

STATELESS_EDGE = 10_000  # D1-stateless: 10 s cold TTFT edge at 10,000 tokens (MEASURED c2246b1f)


@dataclass(frozen=True)
class Opts:
    name: str
    prec: str = "i8"
    m: float = 0.0
    slo: bool = True
    reprefill: bool = False
    excursion: bool = False
    filt: bool = False  # client tool-output filter (keep 0.25)
    compact: bool = False  # compaction at first cloud contact (keep 0.25, executor priced)
    reset_on_migrate: bool = True  # D1/D3: migration sets cached = 0
    waste_on_migrate_s: float = 0.0  # D1/D3: the 10 s timeout that triggered the migration
    stateless_edge: int | None = None
    all_cloud: bool = False
    nnew_blind: bool = False  # ablation: predicted TTFT ignores the delta size
    kappa_blind: bool = False  # ablation: admission by memory only


@dataclass
class DResult:
    name: str
    usd: float = 0.0
    first_cloud_turn: int | None = None
    local_turns: int = 0
    excursions: int = 0
    waste_s: float = 0.0
    slo_violations: int = 0  # admitted local turns whose true TTFT > SLO
    trace: list[str] = field(default_factory=list)


def run(turns: list[DTurn], price: prm.Price, o: Opts) -> DResult:
    led = Ledger(price)
    r = DResult(o.name)
    pc = 0
    pd = 0
    migrated = False
    contacted = False

    def first_contact() -> None:
        nonlocal pc, contacted
        if not contacted and o.compact:
            pc = led.compact(pc)
        contacted = True

    for t, turn in enumerate(turns):
        tool = int(prm.FILTER_KEEP * turn.tool) if o.filt else turn.tool
        dnew = turn.new + tool
        if o.all_cloud or migrated:
            pc += dnew
            led.call(pc, turn.out)
            r.first_cloud_turn = r.first_cloud_turn or t + 1
            r.trace.append(f"t{t + 1} cloud P_c={pc}")
            continue
        p = pd + dnew
        if o.stateless_edge is not None:
            ok = p <= o.stateless_edge
            t_loc = cold(p)
        else:
            t_loc = cold(p) if t == 0 else ext(dnew, p)
            if o.reprefill and cold_mem_ok(p, o.prec):
                t_loc = min(t_loc, cold(p))
            t_pred = t_loc
            if o.nnew_blind and t > 0:
                t_pred = ext(prm.SMALL_TURN, p)
            mem_ok = p <= session_cap(o.prec, dnew, o.m)
            ok = mem_ok and (not o.slo or o.kappa_blind or t_pred <= prm.SLO_S)
            if ok and o.nnew_blind and t_loc > prm.SLO_S:
                ok = False  # true TTFT breaks the SLO: timeout, then migrate like D1
                r.waste_s += prm.SLO_S
            if ok and o.kappa_blind and t_loc > prm.SLO_S:
                r.slo_violations += 1
        if ok:
            pd = p
            pc += dnew
            r.local_turns += 1
            r.trace.append(f"t{t + 1} local P={p} ttft={t_loc:.2f}")
            continue
        if o.excursion and turn.tool > 0:
            pstub = pd + turn.new + prm.STUB_TOKENS
            nxt = pstub + prm.SMALL_TURN
            if (
                pstub <= session_cap(o.prec, prm.SMALL_TURN, o.m)
                and ext(prm.SMALL_TURN, nxt) <= prm.SLO_S
            ):
                first_contact()
                pc += dnew
                led.call(pc, turn.out)
                pd = pstub
                r.excursions += 1
                r.first_cloud_turn = r.first_cloud_turn or t + 1
                r.trace.append(f"t{t + 1} excursion P_c={pc} P_d={pd}")
                continue
        first_contact()
        if o.reset_on_migrate:
            led.reset()
        pc += dnew
        led.call(pc, turn.out)
        migrated = True
        r.waste_s += o.waste_on_migrate_s
        r.first_cloud_turn = r.first_cloud_turn or t + 1
        r.trace.append(f"t{t + 1} migrate P={p} ttft={t_loc:.2f} P_c={pc}")
    r.usd = led.usd
    return r


D1_KW = {"waste_on_migrate_s": prm.SLO_S}
DUET_HW = {"m": prm.MARGIN_M, "reprefill": True, "excursion": True, "reset_on_migrate": False}

POLICIES: dict[str, Opts] = {
    "D0": Opts("D0", all_cloud=True),
    "D1-stateless": Opts("D1-stateless", stateless_edge=STATELESS_EDGE),
    "D1-resident": Opts("D1-resident", **D1_KW),  # type: ignore[arg-type]
    "D3": Opts("D3", filt=True, compact=True, **D1_KW),  # type: ignore[arg-type]
    "DUET-HW": Opts("DUET-HW", **DUET_HW),  # type: ignore[arg-type]
    "DUET-HW-excursion-only": Opts(
        "DUET-HW-excursion-only", m=prm.MARGIN_M, excursion=True, reset_on_migrate=False
    ),
    "DUET-HW-reprefill-only": Opts(
        "DUET-HW-reprefill-only", m=prm.MARGIN_M, reprefill=True, reset_on_migrate=False
    ),
    "DUET-full": Opts("DUET-full", filt=True, compact=True, **DUET_HW),  # type: ignore[arg-type]
    "config-only-bg": Opts("config-only-bg", prec="u4", m=prm.MARGIN_M, slo=False),
    "D1-bg": Opts("D1-bg", slo=False),
    "D3-bg": Opts("D3-bg", slo=False, filt=True, compact=True),
    "DUET-bg": Opts("DUET-bg", prec="u4", m=prm.MARGIN_M, slo=False, filt=True, compact=True),
    "kappa-blind": Opts("kappa-blind", kappa_blind=True),
    "nnew-blind": Opts("nnew-blind", nnew_blind=True),
}


def simulate(name: str, turns: list[DTurn], price: prm.Price = prm.SONNET) -> DResult:
    return run(turns, price, POLICIES[name])
