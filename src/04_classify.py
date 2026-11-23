"""Extract package mentions from all generations and classify each unique name.

Outputs (data/derived/):
  mentions.parquet  one row per (model, lang, set, idx, source, name)
  names.parquet     one row per (ecosystem, name) with frequency features and 2024/now status
  candidates.txt    names absent from the 2024 snapshot (input to 05_enrich)
"""
from __future__ import annotations

import json

import pandas as pd

from common import ALL_MODELS, DERIVED, GEN, REGISTRY
from extract import extract_row

ECO = {"Python": "pypi", "Javascript": "npm"}


def load_mentions() -> pd.DataFrame:
    rows = []
    for model, meta in ALL_MODELS.items():
        path = GEN / (model.replace(":", "__").replace("/", "_") + ".jsonl")
        if not path.exists():
            print("no generations yet:", model)
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for src, name, dotted in extract_row(r):
                    rows.append({"model": model, "era": meta["era"], "cohort": meta["cohort"], "lang": r["lang"], "eco": ECO[r["lang"]],
                                 "set": r["set"], "idx": r["idx"], "source": src, "name": name, "dotted": dotted})
    return pd.DataFrame(rows)


def main() -> None:
    m = load_mentions()
    m.to_parquet(DERIVED / "mentions.parquet", index=False)
    py24 = set((REGISTRY / "pypi_2024.txt").read_text(encoding="utf-8").split("\n"))
    npm24 = set((REGISTRY / "npm_2024.txt").read_text(encoding="utf-8").split("\n"))
    pynow = set((REGISTRY / "pypi_now.txt").read_text(encoding="utf-8").split("\n"))

    g = m.groupby(["eco", "name"])
    names = pd.DataFrame({
        "mentions": g.size(),
        "n_models": g["model"].nunique(),
        "n_models_2024": g.apply(lambda d: d.loc[d.cohort == "2024-local", "model"].nunique()),
        "n_models_2026": g.apply(lambda d: d.loc[d.cohort == "2026-local", "model"].nunique()),   # small local models
        "n_models_2026h": g.apply(lambda d: d.loc[d.cohort == "2026-hosted", "model"].nunique()),  # large hosted models
        "n_prompts": g.apply(lambda d: d[["lang", "set", "idx"]].drop_duplicates().shape[0]),
        "sources": g["source"].agg(lambda s: ",".join(sorted(set(s)))),
        "only_dotted": g["dotted"].all(),       # every mention was a module path like 'matplotlib.pyplot'
    }).reset_index()
    names["in_2024"] = [(n in py24) if e == "pypi" else (n in npm24) for e, n in zip(names.eco, names.name)]
    names["exists_now_list"] = [(n in pynow) if e == "pypi" else None for e, n in zip(names.eco, names.name)]
    names.to_parquet(DERIVED / "names.parquet", index=False)

    cand = names[~names.in_2024]
    (DERIVED / "candidates.txt").write_text("\n".join(f"{e}\t{n}" for e, n in zip(cand.eco, cand.name)), encoding="utf-8")

    print("mentions:", len(m), "| unique names:", len(names))
    print(names.groupby("eco").agg(unique=("name", "size"), absent_2024=("in_2024", lambda s: int((~s).sum()))))
    per_model = (m.assign(h=~m.set_index(["eco", "name"]).index.isin(names[names.in_2024].set_index(["eco", "name"]).index))
                 .groupby(["model", "eco"])["h"].mean().unstack().round(3))
    print("share of mentions absent from 2024 snapshot, by model:\n", per_model)


if __name__ == "__main__":
    main()
