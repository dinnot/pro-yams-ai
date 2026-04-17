#!/usr/bin/env python3
"""Extract summary metrics from a finished run."""
import sys, pathlib, csv, json

run_dir = pathlib.Path(sys.argv[1])

def read_csv(p):
    if not p.exists(): return []
    with open(p) as f:
        return list(csv.DictReader(f))

tl = read_csv(run_dir / "training_log.csv")
el = read_csv(run_dir / "eval_log.csv")

out = {"run": str(run_dir.name)}
if tl:
    last = tl[-1]
    out["step_last"] = int(last["step"])
    out["loss_last"] = float(last["loss"])
    out["gps_last"]  = float(last["gps"])
    out["games_played"] = int(last["games_played"])
    out["temp_last"] = float(last["temperature"])
if el:
    last = el[-1]
    out["eval_step_last"] = int(last["step"])
    out["win_rate_last"]  = float(last["win_rate"])
    out["margin_last"]    = float(last["avg_margin"])
    out["eval_curve"] = [(int(r["step"]), float(r["win_rate"]), float(r["avg_margin"])) for r in el]

print(json.dumps(out, indent=2))
