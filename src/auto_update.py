"""Keeps the derived data current while generation runs, then runs the full analysis once.

Every --every minutes: if any generation file grew, run 04_classify -> 05_enrich -> 06_controls (discovery,
incremental) so registry lookups are spread over the run instead of piling up at the end.
When every local model (2024 discovery + extension, 2026 local) has all prompts: build the C1 controls
(06_controls --set c1), then run 07, 09, 10, 11, 12, 13, 14 and 08, and write data/derived/analysis.finished.
The hosted cohort is not waited for (rate-limited API); rerun 14 when it completes.
Usage: python -u src/auto_update.py [--every 90]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

from common import ALL_MODELS, DERIVED, GEN, MODELS

N_PROMPTS = 3000
SRC = __import__("pathlib").Path(__file__).resolve().parent


def counts() -> dict:
    out = {}
    for m in ALL_MODELS:
        p = GEN / (m.replace(":", "__").replace("/", "_") + ".jsonl")
        out[m] = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
    return out


def step(script: str, *args: str) -> bool:
    log = DERIVED / f"auto_{script.replace('.py', '')}.log"
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as f:
        f.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M')} {script} {' '.join(args)}\n")
        f.flush()
        rc = subprocess.run([sys.executable, "-u", str(SRC / script), *args], cwd=SRC, stdout=f, stderr=subprocess.STDOUT).returncode
    print(time.strftime("%H:%M"), f"{script} {' '.join(args)} -> exit {rc} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return rc == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=90, help="minutes between incremental passes")
    a = ap.parse_args()
    done_marker = DERIVED / "analysis.finished"
    last = None
    while not done_marker.exists():
        c = counts()
        local_done = all(c[m] >= N_PROMPTS for m in MODELS)
        if c != last:
            print(time.strftime("%H:%M"), "incremental pass;", sum(c.values()), "rows", flush=True)
            ok = step("04_classify.py") and step("05_enrich.py") and step("06_controls.py", "--set", "discovery")
            last = c if ok else last
        if local_done and last == c:
            print(time.strftime("%H:%M"), "all local models complete: full analysis", flush=True)
            if not step("06_controls.py", "--set", "c1"):
                time.sleep(600)
                continue
            for s in ("07_stats.py", "09_defense.py", "10_mechanism.py", "11_exposure.py", "12_claimants.py",
                      "13_confounds.py", "14_confirmatory.py", "08_figures.py"):
                step(s)
            done_marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))
            print(time.strftime("%H:%M"), "analysis finished; see tables/confirmatory.csv", flush=True)
            break
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
