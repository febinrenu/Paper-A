"""Robustness, paired design: re-run prompts generated before the server-setting change under the later settings.

DeepSeek-Coder-6.7B generated part of its prompts with Ollama defaults and the rest after the change to 3 parallel
slots, flash attention and a q8_0 KV cache (docs/run_notes.md). The before/after halves contain different prompts,
so here a random sample of the *before* prompts is generated again with the current settings, the same seeds and
the same three-step protocol, and the two runs are compared prompt by prompt.

Usage: python src/20_settings_rerun.py [--n 450]      (resumable; then prints and writes the comparison)
Outputs: data/gen_rerun/deepseek-coder__6.7b-instruct.jsonl, tables/u8_settings_rerun.csv
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json

import httpx
import numpy as np
import pandas as pd

from common import DATA, GEN, REGISTRY, TABLES
from extract import extract_row

gen = importlib.import_module("02_generate")
MODEL = "deepseek-coder:6.7b-instruct"
SWITCH = pd.Timestamp("2026-11-23 15:15")
OUT = DATA / "gen_rerun" / "deepseek-coder__6.7b-instruct.jsonl"
ECO = {"Python": "pypi", "Javascript": "npm"}


def load(path) -> dict:
    rows = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows[(r["lang"], r["set"], r["idx"])] = r
    return rows


async def rerun(sample: list[dict], prompts: dict, concurrency: int = 3) -> None:
    OUT.parent.mkdir(exist_ok=True)
    done = load(OUT)
    todo = [r for r in sample if (r["lang"], r["set"], r["idx"]) not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)
    sem, lock, n = asyncio.Semaphore(concurrency), asyncio.Lock(), 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
        async def one(r):
            nonlocal n
            async with sem:
                try:
                    res = await gen.run_one(c, MODEL, prompts[(r["lang"], r["set"], r["idx"])], None, gen.SAMPLE_SEED)
                except Exception as ex:  # noqa: BLE001
                    print("failed:", r["lang"], r["set"], r["idx"], str(ex)[:120], flush=True)
                    return
            async with lock:
                with open(OUT, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n += 1
                if n % 25 == 0:
                    print(f"  {n}/{len(todo)}", flush=True)
        await asyncio.gather(*(one(r) for r in todo))


def compare(orig: dict, new: dict) -> pd.DataFrame:
    snap = {"pypi": set((REGISTRY / "pypi_2024.txt").read_text(encoding="utf-8").split("\n")),
            "npm": set((REGISTRY / "npm_2024.txt").read_text(encoding="utf-8").split("\n"))}
    per = []
    for k in sorted(set(orig) & set(new)):
        eco = ECO[k[0]]
        a = {n for _, n, _ in extract_row(orig[k])}
        b = {n for _, n, _ in extract_row(new[k])}
        per.append({"key": k, "eco": eco, "n_before": len(a), "n_after": len(b),
                    "h_before": sum(x not in snap[eco] for x in a), "h_after": sum(x not in snap[eco] for x in b),
                    "same_code": orig[k]["code"] == new[k]["code"],
                    "jaccard": len(a & b) / len(a | b) if a | b else 1.0})
    p = pd.DataFrame(per)
    rng = np.random.default_rng(1)
    rows = []
    for eco, g in [("pooled", p)] + list(p.groupby("eco")):
        hb, ha = g.h_before.sum() / max(g.n_before.sum(), 1), g.h_after.sum() / max(g.n_after.sum(), 1)
        boots = []
        for _ in range(2000):
            s = g.iloc[rng.integers(0, len(g), len(g))]
            boots.append(s.h_after.sum() / max(s.n_after.sum(), 1) - s.h_before.sum() / max(s.n_before.sum(), 1))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"eco": eco, "prompts": len(g), "names_before": int(g.n_before.sum()), "names_after": int(g.n_after.sum()),
                     "hall_share_before": hb, "hall_share_after": ha, "diff": ha - hb, "diff_lo": lo, "diff_hi": hi,
                     "identical_code_share": g.same_code.mean(), "mean_jaccard": g.jaccard.mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    orig = load(GEN / "deepseek-coder__6.7b-instruct.jsonl")
    before = [r for r in orig.values() if pd.Timestamp(r["ts"]) < SWITCH]
    rng = np.random.default_rng(20261126)
    sample = [before[i] for i in sorted(rng.choice(len(before), size=min(a.n, len(before)), replace=False))]
    prompts = {(r["lang"], r["set"], r["idx"]): r for r in gen.load_prompts(375)}
    if not a.compare_only:
        asyncio.run(rerun(sample, prompts))
    keys = {(r["lang"], r["set"], r["idx"]) for r in sample}
    out = compare({k: v for k, v in orig.items() if k in keys}, load(OUT))
    out.to_csv(TABLES / "u8_settings_rerun.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
