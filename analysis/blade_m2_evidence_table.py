import json, os, re, statistics as st, datetime as dt, glob, csv
R = r"C:\Users\rithw\OneDrive\Documents\GitHub\APU\results"
TS = "20260925T043921Z"
rows = [json.loads(l) for l in open(f"{R}\\blade_m2_host_interference_{TS}.jsonl")]
log = open(r"C:\apu\m2_run.log", encoding="utf-8", errors="replace").read().splitlines()
call_end = {}
for l in log:
    m = re.match(r"\[(\S+)\]\s+\[(\S+)\] call=(\d+)", l)
    if m:
        t = dt.datetime.fromisoformat(m.group(1)).timestamp()
        call_end[(m.group(2), int(m.group(3)))] = t

def dmon(tag):
    p = f"{R}\\m2_dmon_{tag}_{TS}.txt"
    lines = [l.split() for l in open(p) if l.strip() and not l.startswith("#") and len(l.split())>=19]
    N = len(lines)
    mt = os.path.getmtime("C:/apu/" + os.path.basename(p))
    out = []
    for i, f in enumerate(lines):
        t = mt - (N - 1 - i)
        out.append((t, dict(pwr=float(f[1]), sm=float(f[4]), pclk=float(f[11]), fb=float(f[14]),
                            rx=float(f[17]), tx=float(f[18]))))
    return out

def med(x):
    return st.median(x) if x else None

tags = ["A_none", "A_memcpy16", "A_spin16", "B_memcpy16_t1", "B_memcpy16_t8",
        "C_dose_5gbps", "C_dose_10gbps", "C_dose_15gbps", "C_dose_20gbps"]
print("tag | n | med TTFT | med decode | med util% | med SMclk | med pwrW | med rx | med tx | active_rows | maxShared(MiB) | maxFB")
res = {}
for tag in tags:
    rs = [r for r in rows if r["tag"] == tag]
    d = dmon(tag)
    act = []
    for r in rs:
        te = call_end[(tag, r["call"])]
        ts_ = te - r["total_s"]
        act += [x for (t, x) in d if ts_ - 1 <= t <= te + 1]
    p = f"{R}\\m2_gpumem_{tag}_{TS}.csv"
    sh = []
    for row in csv.DictReader(open(p)):
        if row["gpu_shared_usage"]:
            sh.append(int(row["gpu_shared_usage"]))
    shm = f"{max(sh)/2**20:.0f}" if sh else "NOT CAPTURED"
    g = lambda k: med([x[k] for x in act])
    res[tag] = dict(ttft=med([r["ttft_s"] for r in rs]), dec=med([r["decode_tps"] for r in rs]),
                    util=g("sm"), clk=g("pclk"), pwr=g("pwr"), rx=g("rx"), tx=g("tx"))
    print(tag, "|", len(rs), "|", round(res[tag]["ttft"], 2), "|", round(res[tag]["dec"], 2), "|",
          g("sm"), "|", g("pclk"), "|", g("pwr"), "|", g("rx"), "|", g("tx"), "|", len(act), "|", shm, "|",
          max(x["fb"] for x in act))
b = res["A_none"]
print()
for tag in tags[1:]:
    r = res[tag]
    print(tag, "TTFT x%.2f" % (r["ttft"] / b["ttft"]), "decode x%.2f (slowdown)" % (b["dec"] / r["dec"]),
          "util delta %.0f pts" % (r["util"] - b["util"]), "clk %.1f%% of base" % (100 * r["clk"] / b["clk"]),
          "pwr %.0f%% of base" % (100 * r["pwr"] / b["pwr"]), "rx %.0f%% tx %.0f%%" % (100 * r["rx"] / b["rx"] if b["rx"] else -1, 100 * r["tx"] / b["tx"] if b["tx"] else -1))

