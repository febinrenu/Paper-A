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


def build(target: pd.DataFrame, emitted: set) -> pd.DataFrame:
    """The original procedure: one Random(SEED) stream, PyPI then npm, token pool from a 300k snapshot sample."""
    rng = random.Random(SEED)
    rows = []
    for eco in ("pypi", "npm"):
        snap = set((REGISTRY / f"{'pypi' if eco == 'pypi' else 'npm'}_2024.txt").read_text(encoding="utf-8").split("\n"))
        pool = token_pool(rng.sample(sorted(snap), min(300_000, len(snap))))
        forbidden = snap | {n for e, n in emitted if e == eco}
        for n in target.loc[target.eco == eco, "name"]:
            for cnm in make_controls(n, pool, rng, forbidden, K):
                rows.append({"eco": eco, "name": cnm, "matched_to": n})
    return pd.DataFrame(rows, columns=["eco", "name", "matched_to"]).drop_duplicates(["eco", "name"])


def lookup_status(ctl: pd.DataFrame) -> pd.DataFrame:
    pynow = set((REGISTRY / "pypi_now.txt").read_text(encoding="utf-8").split("\n"))
    # PyPI: only names on the current list need a metadata lookup; npm: every control is looked up.
    lookup = [(e, n) for e, n in zip(ctl.eco, ctl.name) if e == "npm" or n in pynow]
    enrich.CACHE = DERIVED / "controls_registry_cache.jsonl"
    asyncio.run(enrich.run(lookup))
    recs = {}
    for line in enrich.CACHE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        recs[(d["eco"], d["name"])] = d
    ctl = ctl.copy()
    ctl["exists"] = [bool(recs.get((e, n), {}).get("exists")) for e, n in zip(ctl.eco, ctl.name)]
    ctl["created"] = [recs.get((e, n), {}).get("created") for e, n in zip(ctl.eco, ctl.name)]
    ctl["tombstone"] = [bool(recs.get((e, n), {}).get("tombstone")) for e, n in zip(ctl.eco, ctl.name)]
    return ctl


def main() -> None:
    """--set discovery: controls for names from the five discovery models. Controls already built are kept
                        unchanged (results were reported from them); names without controls yet (e.g. the end of
                        a run that was still going) get controls from the same procedure applied to those names.
       --set c1:        controls for confirmatory set C1 (docs/analysis_plan.md): extension-model names that no
                        discovery model emitted; the same procedure with a fresh Random(SEED), as pre-specified."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["discovery", "c1"], default="discovery")
    a = ap.parse_args()
    names = pd.read_parquet(DERIVED / "names.parquet")
    emitted = set(zip(names.eco, names.name))
    if a.set == "discovery":
        out = DERIVED / "controls.parquet"
        target = names[(~names.in_2024) & (names.n_models_2024d > 0)]
        old = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=["eco", "name", "matched_to"])
        have = set(zip(old.eco, old.matched_to))
        tried_p = DERIVED / "controls_attempted.txt"            # names that got no control are not retried forever
        tried = set(tuple(l.split("\t")) for l in tried_p.read_text(encoding="utf-8").splitlines() if l) if tried_p.exists() else set()
        missing = target[[(e, n) not in have and (e, n) not in tried for e, n in zip(target.eco, target.name)]]
        print("discovery targets:", len(target), "| already controlled:", len(target) - len(missing), "| new:", len(missing))
        new = build(missing, emitted)
        with open(tried_p, "a", encoding="utf-8") as f:
            f.write("".join(f"{e}\t{n}\n" for e, n in zip(missing.eco, missing.name)))
        new = new[~new.set_index(["eco", "name"]).index.isin(old.set_index(["eco", "name"]).index)]
        ctl = pd.concat([old[["eco", "name", "matched_to"]], new], ignore_index=True)
    else:
        out = DERIVED / "controls_c1.parquet"
        target = names[(~names.in_2024) & (names.n_models_2024x > 0) & (names.n_models_2024d == 0)]
        print("C1 targets:", len(target))
        ctl = build(target, emitted)
    print("controls:", ctl.groupby("eco").size().to_dict())
    ctl = lookup_status(ctl)
    ctl.to_parquet(out, index=False)
    print(ctl.groupby("eco")[["exists", "tombstone"]].mean())


if __name__ == "__main__":
    main()
