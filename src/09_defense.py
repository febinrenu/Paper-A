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

from common import DERIVED, TABLES  # noqa: F401  (DERIVED used by probe_budget)

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


def probe_budget(h24: pd.DataFrame, reps: int = 50, seed: int = 1122, sec_per_prompt: float = 4.0) -> pd.DataFrame:
    """A registry in early 2024 runs m of the 2024-era models on n prompts each and reserves every emitted name
    absent from the registry. How many later-malicious / later-claimed names would that have pre-empted?"""
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")
    m24 = mentions[mentions.era == 2024].merge(h24[["eco", "name", "claimed", "mal_post"]], on=["eco", "name"])
    models = sorted(m24.model.unique())
    prompts = {m: m24[m24.model == m][["lang", "set", "idx"]].drop_duplicates() for m in models}
    tot_mal = h24.groupby("eco").mal_post.sum().to_dict()
    tot_cl = h24.groupby("eco").claimed.sum().to_dict()
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(1, len(models) + 1):
        for n in (100, 300, 1000, 3000):
            for _ in range(reps if (m < len(models) or n < 3000) else 1):
                pick = rng.choice(models, m, replace=False)
                parts = []
                for mod in pick:
                    p = prompts[mod]
                    sel = p.sample(min(n, len(p)), random_state=int(rng.integers(1 << 31)))
                    parts.append(m24[m24.model == mod].merge(sel, on=["lang", "set", "idx"]))
                w = pd.concat(parts).drop_duplicates(["eco", "name"])
                for eco, g in w.groupby("eco"):
                    rows.append({"eco": eco, "models": m, "prompts_per_model": n, "gpu_hours": m * n * sec_per_prompt / 3600,
                                 "watchlist": len(g), "malicious_caught": int(g.mal_post.sum()),
                                 "malicious_recall": g.mal_post.sum() / max(tot_mal.get(eco, 0), 1),
                                 "claimed_caught": int(g.claimed.sum()),
                                 "claimed_recall": g.claimed.sum() / max(tot_cl.get(eco, 0), 1)})
    df = pd.DataFrame(rows)
    return df.groupby(["eco", "models", "prompts_per_model"]).agg(
        gpu_hours=("gpu_hours", "first"), watchlist=("watchlist", "mean"), malicious_caught=("malicious_caught", "mean"),
        malicious_recall=("malicious_recall", "mean"), malicious_recall_sd=("malicious_recall", "std"),
        claimed_recall=("claimed_recall", "mean")).reset_index()


def main() -> None:
    mech = importlib.import_module("10_mechanism")
    h24 = mech.hallucinated(st.CUTOFF)
    h24["malicious"] = h24.mal_post
    pb = probe_budget(h24)
    pb.to_csv(TABLES / "rq5_probe_budget.csv", index=False)
    print(pb[pb.prompts_per_model.isin([300, 3000])].round(3).to_string(index=False))
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
            dm = d.assign(claimed=d.malicious)          # same ranking, malicious claims as the target
            summ.append({"eco": eco, "policy": name, "candidates": n, "claims": int(d.claimed.sum()),
                         "malicious": int(d.malicious.sum()),
                         "K_for_50pct": k_for(d, s, 0.5), "K_for_80pct": k_for(d, s, 0.8),
                         "K_for_50pct_malicious": k_for(dm, s, 0.5), "K_for_80pct_malicious": k_for(dm, s, 0.8)})
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
