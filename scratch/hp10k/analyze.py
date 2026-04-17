#!/usr/bin/env python3
"""Read all hp10k runs and print a sorted table."""
import os, csv, json, pathlib, sys

ROOT = pathlib.Path("/home/sorin/pro_yams_ai/logs/hp10k")

def read_csv(p):
    if not p.exists(): return []
    with open(p) as f: return list(csv.DictReader(f))

results = []
for d in sorted(ROOT.iterdir()):
    if not d.is_dir(): continue
    tl = read_csv(d / "training_log.csv")
    el = read_csv(d / "eval_log.csv")
    if not el: continue
    last_el = el[-1]
    last_tl = tl[-1] if tl else {}
    results.append({
        "run": d.name,
        "step": int(last_el["step"]),
        "wr": float(last_el["win_rate"]),
        "margin": float(last_el["avg_margin"]),
        "wr2k": float(el[0]["win_rate"]) if el else 0,
        "wr4k": float(el[1]["win_rate"]) if len(el)>=2 else 0,
        "wr6k": float(el[2]["win_rate"]) if len(el)>=3 else 0,
        "wr8k": float(el[3]["win_rate"]) if len(el)>=4 else 0,
        "gps": float(last_tl["gps"]) if last_tl else 0.0,
        "loss": float(last_tl["loss"]) if last_tl else 0.0,
        "games": int(last_tl["games_played"]) if last_tl else 0,
    })

# Sort by win rate descending
results.sort(key=lambda r: -r["wr"])

print(f"{'run':<20} {'step':>5} {'wr2k':>6} {'wr4k':>6} {'wr6k':>6} {'wr8k':>6} {'wr10k':>6} {'margin':>8} {'gps':>7} {'loss':>9} {'games':>8}")
print("-" * 115)
for r in results:
    print(f"{r['run']:<20} {r['step']:>5} {r['wr2k']:>6.3f} {r['wr4k']:>6.3f} {r['wr6k']:>6.3f} {r['wr8k']:>6.3f} {r['wr']:>6.3f} {r['margin']:>8.1f} {r['gps']:>7.1f} {r['loss']:>9.5f} {r['games']:>8d}")
