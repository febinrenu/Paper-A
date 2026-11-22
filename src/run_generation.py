"""Unattended overnight driver: runs every cohort model, one at a time on the GPU.

Picks the next model that is downloaded and not yet complete (cohort order as tie-break), so
the GPU never idles waiting for one slow download. Waits for any externally started run
(--after) to finish first. Exits when all models are complete or nothing new arrives for 12 h.
Usage: python src/run_generation.py [--after codellama:7b-instruct] [--per-set 375]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import GEN, MODELS, ROOT


def available() -> set:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags") as r:
            names = {m["name"] for m in json.load(r)["models"]}
    except OSError:
        return set()
    return names | {n.removesuffix(":latest") for n in names}


def n_rows(model: str) -> int:
    p = GEN / (model.replace(":", "__").replace("/", "_") + ".jsonl")
    return sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default="")
    ap.add_argument("--per-set", type=int, default=375)
    a = ap.parse_args()
    expected = a.per_set * 8
    if a.after:
        while n_rows(a.after) < expected:
            time.sleep(120)
    idle_since = time.time()
    while True:
        pending = [m for m in MODELS if n_rows(m) < expected]
        if not pending:
            print("all models complete", flush=True)
            return
        ready = [m for m in pending if m in available()]
        if not ready:
            if time.time() - idle_since > 12 * 3600:
                print("no model became available for 12 h; pending:", pending, flush=True)
                return
            time.sleep(120)
            continue
        m = ready[0]
        print(time.strftime("%Y-%m-%d %H:%M"), "starting", m, f"({n_rows(m)}/{expected} done)", flush=True)
        subprocess.run([sys.executable, "-u", str(ROOT / "src/02_generate.py"), "--models", m,
                        "--per-set", str(a.per_set), "--concurrency", "4"], check=False)
        if n_rows(m) < expected:      # crashed/aborted: avoid a tight loop, retry later
            time.sleep(300)
        idle_since = time.time()


if __name__ == "__main__":
    main()
