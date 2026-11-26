"""Hashed watchlist of hallucinated package names that were still unregistered at the data freeze.

Each entry is SHA-256(SALT + "<eco>:<normalised name>") in hex, one per line, sorted. The salt is published with
the file, so anyone can test a given name (registries before accepting a new project, install-time tools before
resolving a dependency) but the file itself does not list names. The clear-text list is shared only with the PyPI
and npm security teams. Names are normalised as the registries do: PyPI per PEP 503 (lower case, runs of -_. to -),
npm lower case.

Usage:  python src/21_watchlist.py                 build release/watchlist_sha256.txt (+ README)
        python src/21_watchlist.py --check pypi NAME   test one name against the built file
Outputs: release/watchlist_sha256.txt, release/WATCHLIST.md, tables/u9_watchlist.csv (counts only)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import re
import sys

import pandas as pd

from common import ROOT, TABLES

st = importlib.import_module("07_stats")
SALT = "slopsquat-watchlist-v1:"
OUT = ROOT / "release"


def norm(eco: str, name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower() if eco == "pypi" else name.lower()


def h(eco: str, name: str) -> str:
    return hashlib.sha256(f"{SALT}{eco}:{norm(eco, name)}".encode()).hexdigest()


def build() -> None:
    cand, _ = st.load_candidates(st.CUTOFF)
    u = cand[cand.status == "unregistered"]
    hashes = sorted({h(e, n) for e, n in zip(u.eco, u.name)})
    OUT.mkdir(exist_ok=True)
    (OUT / "watchlist_sha256.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")
    counts = u.groupby("eco").size().rename("names").reset_index()
    counts.to_csv(TABLES / "u9_watchlist.csv", index=False)
    (OUT / "WATCHLIST.md").write_text(
        "# Hashed watchlist of hallucinated package names\n\n"
        f"{len(hashes)} names that language models recommended in our runs and that were not registered on PyPI or "
        "npm at the data freeze (2026-11-23).\n\n"
        f"Each line is `sha256(\"{SALT}\" + eco + \":\" + name)` in hex, where `eco` is `pypi` or `npm` and `name` is "
        "normalised (PyPI: lower case, runs of `-_.` replaced by `-`; npm: lower case).\n\n"
        "```python\nimport hashlib\n"
        f"def listed(eco, name, path='watchlist_sha256.txt', salt='{SALT}'):\n"
        "    import re\n"
        "    n = re.sub(r'[-_.]+', '-', name).lower() if eco == 'pypi' else name.lower()\n"
        "    d = hashlib.sha256(f'{salt}{eco}:{n}'.encode()).hexdigest()\n"
        "    return d in set(open(path).read().split())\n```\n\n"
        "A match means the name was invented by at least one model in our study; registering or depending on it "
        "deserves scrutiny. Absence means nothing: every model invents names we did not observe.\n",
        encoding="utf-8")
    print(f"{len(hashes)} hashes -> {OUT / 'watchlist_sha256.txt'}")
    print(counts.to_string(index=False))


def check(eco: str, name: str) -> None:
    hs = set((OUT / "watchlist_sha256.txt").read_text(encoding="ascii").split())
    print("listed" if h(eco, name) in hs else "not listed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=2, metavar=("ECO", "NAME"))
    a = ap.parse_args()
    check(*a.check) if a.check else build()
