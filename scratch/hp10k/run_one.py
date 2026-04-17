#!/usr/bin/env python3
"""Run one 10k-step training experiment.

Usage:  python run_one.py <config.yaml>
"""
import os, sys, subprocess, time, pathlib

ROOT = pathlib.Path("/home/sorin/pro_yams_ai")
BIN  = ROOT / "build" / "pro_yams_ai"

def main():
    cfg_path = pathlib.Path(sys.argv[1]).resolve()
    run_dir  = cfg_path.parent
    stderr   = run_dir / "stderr.log"
    stdout   = run_dir / "stdout.log"

    t0 = time.time()
    with open(stdout, "w") as so, open(stderr, "w") as se:
        r = subprocess.run(
            [str(BIN), "--mode", "train",
             "--debug_mode", "1",
             "--config", str(cfg_path)],
            stdout=so, stderr=se, cwd=str(ROOT))
    dt = time.time() - t0
    ok = (r.returncode == 0)
    print(f"{'OK' if ok else 'FAIL'} in {dt:.1f}s  ret={r.returncode}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
