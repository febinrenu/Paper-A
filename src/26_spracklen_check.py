"""Pipeline validation: per-model hallucination rates against those published by Spracklen et al. (USENIX Sec 2025).

Their figure data (Zenodo artifact, Plots/Data/figure_2.csv: model, Python %, JavaScript %) give the share of
recommended packages that were hallucinated, per model, for the full-precision models with a 2,048-token cap on
their full prompt set. We compare our 4-bit, 1,024-token, 3,000-prompt runs for the six model families both studies
share. Outputs: tables/u13_spracklen_check.csv.
"""
from __future__ import annotations

import pandas as pd
from scipy import stats

from common import RAW, TABLES

MAP = {"CodeLlama 7B": "codellama:7b-instruct", "DeepSeek 6B": "deepseek-coder:6.7b-instruct",
       "DeepSeek 1B": "deepseek-coder:1.3b-instruct", "Mistral": "mistral:7b-instruct-v0.2-q4_K_M",
       "OpenChat": "openchat:7b-v3.5", "Magicoder": "magicoder:7b"}


def main() -> None:
    theirs = pd.read_csv(RAW / "spracklen_plots" / "figure_2.csv", header=None, names=["label", "pypi", "npm"])
    theirs = theirs[theirs.label.isin(MAP)].assign(model=lambda d: d.label.map(MAP))
    ours = pd.read_csv(TABLES / "per_model.csv").pivot(index="model", columns="eco", values="hallucinated_share") * 100
    rows = []
    for r in theirs.itertuples():
        rows.append({"label": r.label, "model": r.model, "theirs_pypi": float(r.pypi), "ours_pypi": ours.loc[r.model, "pypi"],
                     "theirs_npm": float(r.npm), "ours_npm": ours.loc[r.model, "npm"]})
    d = pd.DataFrame(rows)
    a = pd.concat([d.theirs_pypi, d.theirs_npm]).to_numpy()
    b = pd.concat([d.ours_pypi, d.ours_npm]).to_numpy()
    d.attrs = {}
    summ = {"label": "summary", "spearman": stats.spearmanr(a, b).statistic, "pearson": stats.pearsonr(a, b).statistic,
            "mean_abs_diff": abs(a - b).mean(), "mean_diff": (b - a).mean()}
    out = pd.concat([d, pd.DataFrame([summ])], ignore_index=True)
    out.to_csv(TABLES / "u13_spracklen_check.csv", index=False)
    print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
