"""RQ5: could registries have pre-empted the claims? Retrospective evaluation of reservation policies.

Setting: at the cutoff (early 2024) a registry knows the names that 2024-era models hallucinate (e.g. by
running exactly this kind of probing) and reserves the top-K names under some ranking. A name later
claimed by a third party counts as "prevented" if it was in the reserved set.

Rankings (all use only information available at the cutoff, i.e. the 2024 cohort's generations):
  mentions     total times the name was emitted
  n_models     number of distinct 2024 models emitting it (ties broken by mentions)
  n_prompts    number of distinct prompts producing it (ties broken by mentions)
  logit_cv     out-of-fold probability from a logistic model on name features (5-fold, no leakage)
  random       expected value of a random reservation of K names (analytic baseline)

Reported for each K in a grid and for each ecosystem:
  reserved K, claims prevented (recall over all later claims), malicious claims prevented,
  precision (claimed / K). Also the K needed to prevent 50% and 80% of claims.
Output: tables/rq5_defense_curve.csv, tables/rq5_defense_summary.csv (aggregates only).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd

from common import DERIVED, TABLES

st = importlib.import_module("07_stats")


def features(d: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "log_mentions": np.log1p(d.mentions), "n_models": d.n_models_2024, "log_prompts": np.log1p(d.n_prompts),
        "length": d.name.str.len(), "n_tokens": d.name.str.count(r"[-_.]") + 1,
        "dotted": d.only_dotted.astype(int), "scoped": d.name.str.startswith("@").astype(int),
    }, index=d.index)


def logit_oof(d: pd.DataFrame, folds: int = 5, seed: int = 1122) -> pd.Series:
    """Out-of-fold predicted claim probability (plain IRLS logistic regression, no extra dependency)."""
    import statsmodels.api as sm
    X = sm.add_constant(features(d), has_constant="add")
    y = d.claimed.astype(int)
    out = pd.Series(np.nan, index=d.index)
    idx = np.random.default_rng(seed).permutation(len(d))
    for f in range(folds):
        test = d.index[idx[f::folds]]
        train = d.index.difference(test)
        if y[train].sum() < 5:
            out[test] = y[train].mean()
            continue
        try:
            fit = sm.Logit(y[train], X.loc[train]).fit(disp=0, maxiter=200)
            out[test] = fit.predict(X.loc[test])
        except Exception:  # noqa: BLE001  (separation etc.)
            fit = sm.GLM(y[train], X.loc[train], family=sm.families.Binomial()).fit_regularized(alpha=1e-3)
            out[test] = fit.predict(X.loc[test])
    return out


def curve(d: pd.DataFrame, score: pd.Series, ks: list[int]) -> list[dict]:
    order = d.assign(_s=score, _m=d.mentions).sort_values(["_s", "_m"], ascending=False)
    claimed, mal = order.claimed.to_numpy(), order.malicious.to_numpy()
    cc, cm = np.cumsum(claimed), np.cumsum(mal)
    tot_c, tot_m = max(int(claimed.sum()), 1), max(int(mal.sum()), 1)
    rows = []
    for k in ks:
        k = min(k, len(order))
        rows.append({"K": k, "prevented": int(cc[k - 1]), "recall": cc[k - 1] / tot_c,
                     "malicious_prevented": int(cm[k - 1]), "malicious_recall": cm[k - 1] / tot_m if mal.sum() else np.nan,
                     "precision": cc[k - 1] / k})
    return rows


def k_for(d: pd.DataFrame, score: pd.Series, target: float) -> int | None:
    order = d.assign(_s=score, _m=d.mentions).sort_values(["_s", "_m"], ascending=False)
    cc = np.cumsum(order.claimed.to_numpy())
    if cc[-1] == 0:
        return None
    return int(np.searchsorted(cc, target * cc[-1]) + 1)


def main() -> None:
    cand, _ = st.load_candidates(pd.Timestamp(st.CUTOFF))
    h24 = cand[(cand.status != "pre_cutoff") & (cand.n_models_2024 > 0)].copy()
    rows, summ = [], []
    for eco, d in h24.groupby("eco"):
        d = d.reset_index(drop=True)
        n = len(d)
        ks = sorted({k for k in [10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 25000, n] if k <= n})
        scores = {"mentions": d.mentions.astype(float),
                  "n_models": d.n_models_2024 * 1e6 + d.mentions,
                  "n_prompts": d.n_prompts.astype(float)}
        if d.claimed.sum() >= 10:
            scores["logit_cv"] = logit_oof(d)
        for name, s in scores.items():
            for r in curve(d, s, ks):
                rows.append({"eco": eco, "policy": name, "candidates": n, **r})
            summ.append({"eco": eco, "policy": name, "candidates": n, "claims": int(d.claimed.sum()),
                         "malicious": int(d.malicious.sum()),
                         "K_for_50pct": k_for(d, s, 0.5), "K_for_80pct": k_for(d, s, 0.8)})
        base = d.claimed.mean()
        for k in ks:      # random reservation: expected recall = K/n
            rows.append({"eco": eco, "policy": "random", "candidates": n, "K": k, "prevented": base * k,
                         "recall": k / n, "malicious_prevented": d.malicious.mean() * k,
                         "malicious_recall": k / n, "precision": base})
    pd.DataFrame(rows).to_csv(TABLES / "rq5_defense_curve.csv", index=False)
    s = pd.DataFrame(summ)
    s.to_csv(TABLES / "rq5_defense_summary.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
