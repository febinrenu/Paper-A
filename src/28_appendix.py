"""Reproducibility appendix: exact local model weights (Ollama digests) and the counts at each stage of the
name pipeline. Outputs: tables/u15_models.csv, tables/u15_pipeline.csv (aggregates only)."""
from __future__ import annotations

import importlib

import httpx
import pandas as pd

from common import ALL_MODELS, DERIVED, TABLES

st = importlib.import_module("07_stats")


def models() -> None:
    tags = {m["name"]: m for m in httpx.get("http://localhost:11434/api/tags", timeout=30).json()["models"]}
    ver = httpx.get("http://localhost:11434/api/version", timeout=30).json().get("version")
    rows = []
    for name, meta in ALL_MODELS.items():
        if name.startswith("nim:"):
            continue
        t = tags.get(name) or tags.get(name + ":latest")
        rows.append({"model": name, "era": meta["era"], "role": meta.get("role") or ("2026" if meta["era"] == 2026 else ""),
                     "digest": (t or {}).get("digest", "")[:12],
                     "params": ((t or {}).get("details") or {}).get("parameter_size"),
                     "quant": ((t or {}).get("details") or {}).get("quantization_level"), "ollama": ver})
    pd.DataFrame(rows).to_csv(TABLES / "u15_models.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


def pipeline() -> None:
    m = pd.read_parquet(DERIVED / "mentions.parquet")
    m24 = m[m.era == 2024]
    names = pd.read_parquet(DERIVED / "names.parquet")
    n24 = names[names.n_models_2024 > 0]
    cand, _ = st.load_candidates(st.CUTOFF)
    c24 = cand[cand.n_models_2024 > 0]
    rows = []
    for eco in ("npm", "pypi"):
        rows.append({"eco": eco, "mentions": int((m24.eco == eco).sum()),
                     "unique_names": int((n24.eco == eco).sum()),
                     "absent_from_snapshot": int(((n24.eco == eco) & ~n24.in_2024).sum()),
                     "registered_before_cutoff": int(((c24.eco == eco) & (c24.status == "pre_cutoff")).sum()),
                     "eligible": int(((c24.eco == eco) & (c24.status != "pre_cutoff")).sum())})
    d = pd.DataFrame(rows)
    d.to_csv(TABLES / "u15_pipeline.csv", index=False)
    print(d.to_string(index=False))


def token_cap() -> None:
    """Share of 2024-model code answers that stopped at the 1,024-token generation cap."""
    import json
    from common import GEN
    tot = hit = 0
    for name, meta in ALL_MODELS.items():
        if meta["era"] != 2024:
            continue
        with open(GEN / (name.replace(":", "__").replace("/", "_") + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                tot += 1
                hit += r.get("code_done") == "length"
    pd.DataFrame([{"answers": tot, "at_cap": hit, "share": hit / tot}]).to_csv(TABLES / "u15_cap.csv", index=False)


if __name__ == "__main__":
    models()
    pipeline()
    token_cap()
