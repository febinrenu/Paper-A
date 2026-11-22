"""Matched control names: the organic registration baseline.

For every hallucinated name (absent from the 2024 snapshot, emitted by a 2024-era model) we
build K control names with the same structure: split on '-' (scoped npm names keep their
scope), replace ONE token by a different token of similar length drawn with frequency
weighting from tokens of real 2024 package names. Controls must be absent from the 2024
snapshot and never emitted by any model. Their registration rate after the snapshot is the
baseline against which the hallucinated-name registration rate is compared.

Output: data/derived/controls.parquet (control name, matched hallucinated name, registry status)
"""
from __future__ import annotations

import asyncio
import collections
import json
import random

import pandas as pd

from common import DERIVED, REGISTRY
from importlib import import_module

enrich = import_module("05_enrich")
K = 3
SEED = 1122


def token_pool(names: list[str]) -> dict[int, tuple[list[str], list[int]]]:
    cnt = collections.Counter(t for n in names for t in n.split("/")[-1].split("-") if 2 <= len(t) <= 20 and t.isalnum())
    by_len = collections.defaultdict(list)
    for t, c in cnt.items():
        if c >= 5:
            by_len[min(len(t), 12)].append((t, c))
    return {L: ([t for t, _ in v], [c for _, c in v]) for L, v in by_len.items()}


def make_controls(name: str, pool, rng: random.Random, forbidden: set, k: int) -> list[str]:
    scope, base = (name.split("/", 1) if name.startswith("@") and "/" in name else ("", name))
    toks = base.split("-")
    out, tries = [], 0
    while len(out) < k and tries < 200:
        tries += 1
        i = rng.randrange(len(toks))
        L = min(len(toks[i]), 12)
        if L not in pool:
            continue
        words, weights = pool[L]
        new = rng.choices(words, weights)[0]
        if new == toks[i]:
            continue
        cand = "-".join(toks[:i] + [new] + toks[i + 1:])
        cand = f"{scope}/{cand}" if scope else cand
        if cand not in forbidden and cand not in out:
            out.append(cand)
    return out


def main() -> None:
    names = pd.read_parquet(DERIVED / "names.parquet")
    emitted = set(zip(names.eco, names.name))
    target = names[(~names.in_2024) & (names.n_models_2024 > 0)]
    rng = random.Random(SEED)
    rows = []
    for eco in ("pypi", "npm"):
        snap = set((REGISTRY / f"{'pypi' if eco == 'pypi' else 'npm'}_2024.txt").read_text(encoding="utf-8").split("\n"))
        pool = token_pool(rng.sample(sorted(snap), min(300_000, len(snap))))
        forbidden = snap | {n for e, n in emitted if e == eco}
        for n in target.loc[target.eco == eco, "name"]:
            for cnm in make_controls(n, pool, rng, forbidden, K):
                rows.append({"eco": eco, "name": cnm, "matched_to": n})
    ctl = pd.DataFrame(rows).drop_duplicates(["eco", "name"])
    print("controls:", ctl.groupby("eco").size().to_dict())

    pynow = set((REGISTRY / "pypi_now.txt").read_text(encoding="utf-8").split("\n"))
    # PyPI: only names on the current list need a metadata lookup; npm: every control is looked up.
    lookup = [(e, n) for e, n in zip(ctl.eco, ctl.name) if e == "npm" or n in pynow]
    enrich.CACHE = DERIVED / "controls_registry_cache.jsonl"
    asyncio.run(enrich.run(lookup))
    recs = {}
    for line in enrich.CACHE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        recs[(d["eco"], d["name"])] = d
    ctl["exists"] = [bool(recs.get((e, n), {}).get("exists")) for e, n in zip(ctl.eco, ctl.name)]
    ctl["created"] = [recs.get((e, n), {}).get("created") for e, n in zip(ctl.eco, ctl.name)]
    ctl["tombstone"] = [bool(recs.get((e, n), {}).get("tombstone")) for e, n in zip(ctl.eco, ctl.name)]
    ctl.to_parquet(DERIVED / "controls.parquet", index=False)
    print(ctl.groupby("eco")[["exists", "tombstone"]].mean())


if __name__ == "__main__":
    main()
