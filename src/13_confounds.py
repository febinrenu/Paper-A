"""Confounds for the headline (hallucinated names carry later malware more often than controls).

Alternative explanation: attackers pick plausible-looking names, or typo-squat existing packages, and
hallucinated names happen to be more plausible / more typo-like than the token-swap controls.

Two name-level covariates, both computed only from the pre-cutoff (early-2024) registry snapshots:
  plausibility   mean per-character log-probability under a character 5-gram model trained on the
                 2024 snapshot of the same ecosystem (Jelinek-Mercer interpolation of orders 1-5).
                 Neither hallucinated nor control names are in the snapshot, so there is no leakage.
  typo_neighbor  the name is one edit (insert, delete, substitute, adjacent transpose) away from a
                 name registered before the cutoff; for PyPI also one edit from a top-5,000 package
                 (2023 downloads, public ClickHouse dataset) when that list is available.

Analyses (malicious outcome = OSV MAL record published after the cutoff):
  (1) covariate balance: hallucinated vs control
  (2) Mantel-Haenszel odds ratio stratified by ecosystem x plausibility quintile
  (3) the same restricted to names with no typo neighbour
  (4) logistic regression: malicious ~ hallucinated + plausibility + typo_neighbor + length + tokens + eco
Outputs: tables/u7_*.csv (aggregates only), data/derived/name_covariates.parquet (name-level, private).
"""
from __future__ import annotations

import collections
import importlib
import math
import random
import string

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from common import DERIVED, REGISTRY, TABLES

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")

ORDER = 5
LAMBDAS = [0.02, 0.08, 0.15, 0.30, 0.45]          # weights for orders 1..5 (sum 1); fixed a priori
BOS, EOS = "\x02", "\x03"
ALPHABET = string.ascii_lowercase + string.digits + "-_."


def names_2024(eco: str) -> list[str]:
    return [n.strip().lower() for n in (REGISTRY / f"{eco}_2024.txt").read_text(encoding="utf-8").split("\n") if n.strip()]


class CharLM:
    def __init__(self, names: list[str]):
        self.c = [collections.Counter() for _ in range(ORDER + 1)]      # c[k][(ctx, ch)] for context length k
        self.ctx = [collections.Counter() for _ in range(ORDER + 1)]
        for n in names:
            s = BOS * (ORDER - 1) + n + EOS
            for i in range(ORDER - 1, len(s)):
                ch = s[i]
                for k in range(ORDER):
                    ctx = s[i - k:i]
                    self.c[k][(ctx, ch)] += 1
                    self.ctx[k][ctx] += 1
        self.V = len({ch for (_, ch) in self.c[0]}) + 1

    def logp(self, name: str) -> float:
        s = BOS * (ORDER - 1) + name.lower() + EOS
        tot = 0.0
        for i in range(ORDER - 1, len(s)):
            ch, p = s[i], 0.0
            for k in range(ORDER):
                ctx = s[i - k:i]
                d = self.ctx[k].get(ctx, 0)
                pk = self.c[k].get((ctx, ch), 0) / d if d else 0.0
                p += LAMBDAS[k] * pk
            p = 0.999 * p + 0.001 / self.V                               # floor for unseen characters
            tot += math.log(p)
        return tot / (len(s) - ORDER + 1)


def edits1(w: str) -> set:
    splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    out = {a + b[1:] for a, b in splits if b}
    out |= {a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1}
    out |= {a + c + b[1:] for a, b in splits if b for c in ALPHABET}
    out |= {a + c + b for a, b in splits for c in ALPHABET}
    out.discard(w)
    return out


def norm(eco: str, n: str) -> str:
    n = n.lower()
    return n.replace("_", "-").replace(".", "-") if eco == "pypi" else n      # PEP 503 normalisation


def pypi_top(n: int = 5000) -> set | None:
    """Top PyPI projects by 2023 downloads (public ClickHouse demo dataset); None if unreachable."""
    import httpx
    q = (f"SELECT project FROM pypi.pypi_downloads_per_day WHERE date >= '2023-01-01' AND date < '2024-01-01' "
         f"GROUP BY project ORDER BY sum(count) DESC LIMIT {n} FORMAT TabSeparated")
    try:
        r = httpx.post("https://sql-clickhouse.clickhouse.com/?user=demo", content=q, timeout=300)
        r.raise_for_status()
        return {norm("pypi", x) for x in r.text.split("\n") if x.strip()}
    except Exception as ex:  # noqa: BLE001
        print("pypi top list unavailable:", type(ex).__name__)
        return None


def covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    top = pypi_top()
    for eco, g in df.groupby("eco"):
        reg = names_2024(eco)
        rng = random.Random(1122)
        train = reg if len(reg) <= 300_000 else rng.sample(reg, 300_000)      # memory: laptop also runs the GPU job
        lm = CharLM(train)
        regn = {norm(eco, n) for n in reg}
        g = g.copy()
        g["plausibility"] = [lm.logp(n.split("/")[-1]) for n in g.name]
        g["typo_neighbor"] = [bool({norm(eco, e) for e in edits1(n.lower())} & regn) for n in g.name]
        if eco == "pypi" and top is not None:
            g["typo_top"] = [bool({norm(eco, e) for e in edits1(n.lower())} & top) for n in g.name]
        g["length"] = g.name.str.len()
        g["n_tokens"] = g.name.str.split("/").str[-1].str.count(r"[-_.]") + 1
        out.append(g)
        print(eco, "covariates done", len(g), flush=True)
    return pd.concat(out, ignore_index=True)


def mh(d: pd.DataFrame, strata: list[str]) -> dict:
    tabs = []
    for _, g in d.groupby(strata, observed=True):
        a = g[g.halluc]; b = g[~g.halluc]
        t = np.array([[a.mal.sum(), len(a) - a.mal.sum()], [b.mal.sum(), len(b) - b.mal.sum()]], dtype=float)
        if t[:, 0].sum() > 0 and len(a) and len(b):
            tabs.append(t)
    if not tabs:
        return {"strata": 0}
    s = StratifiedTable(tabs)
    lo, hi = s.oddsratio_pooled_confint()
    return {"strata": len(tabs), "or_mh": s.oddsratio_pooled, "or_lo": lo, "or_hi": hi, "p": s.test_null_odds().pvalue,
            "halluc_mal": int(d[d.halluc].mal.sum()), "halluc_n": int(d.halluc.sum()),
            "control_mal": int(d[~d.halluc].mal.sum()), "control_n": int((~d.halluc).sum())}


def main() -> None:
    cutoff = st.CUTOFF
    h, c = mech.hallucinated(cutoff), mech.controls(cutoff)
    d = pd.concat([h[["eco", "name", "mal_post"]].assign(halluc=True), c[["eco", "name", "mal_post"]].assign(halluc=False)],
                  ignore_index=True).rename(columns={"mal_post": "mal"})
    d = d.drop_duplicates(["eco", "name"])
    cache = DERIVED / "name_covariates.parquet"
    if cache.exists():
        cov = pd.read_parquet(cache)
        new = d.merge(cov[["eco", "name"]], on=["eco", "name"], how="left", indicator=True)
        new = new[new._merge == "left_only"].drop(columns="_merge")
        if len(new):
            cov = pd.concat([cov, covariates(new)], ignore_index=True)
    else:
        cov = covariates(d)
    cov.drop(columns=["mal", "halluc"], errors="ignore").to_parquet(cache, index=False)
    d = d.merge(cov.drop(columns=["mal", "halluc"], errors="ignore"), on=["eco", "name"], how="left")

    # (1) balance
    bal = []
    for eco, g in d.groupby("eco"):
        for v in ["plausibility", "typo_neighbor", "typo_top", "length", "n_tokens"]:
            if v not in g or g[v].isna().all():
                continue
            a, b = g[g.halluc][v].astype(float), g[~g.halluc][v].astype(float)
            bal.append({"eco": eco, "variable": v, "halluc_mean": a.mean(), "control_mean": b.mean(),
                        "std_diff": (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2),
                        "mannwhitney_p": stats.mannwhitneyu(a.dropna(), b.dropna()).pvalue})
    bal = pd.DataFrame(bal)
    bal.to_csv(TABLES / "u7_balance.csv", index=False)
    print(bal.round(3).to_string(index=False))

    # (2)/(3) stratified odds ratios
    d["plaus_q"] = d.groupby("eco").plausibility.transform(lambda s: pd.qcut(s, 5, labels=False, duplicates="drop"))
    rows = [{"analysis": "crude (eco strata)", **mh(d, ["eco"])},
            {"analysis": "eco x plausibility quintile", **mh(d, ["eco", "plaus_q"])},
            {"analysis": "no typo neighbour, eco x plausibility quintile", **mh(d[~d.typo_neighbor], ["eco", "plaus_q"])},
            {"analysis": "typo neighbour only, eco x plausibility quintile", **mh(d[d.typo_neighbor], ["eco", "plaus_q"])}]
    if "typo_top" in d:
        p = d[d.eco == "pypi"]
        rows.append({"analysis": "PyPI, not one edit from a top-5k package, plausibility quintile",
                     **mh(p[p.typo_top != True], ["plaus_q"])})  # noqa: E712
    strat = pd.DataFrame(rows)
    strat.to_csv(TABLES / "u7_stratified_or.csv", index=False)
    print(strat.round(4).to_string(index=False))

    # (4) adjusted logistic regression
    import statsmodels.formula.api as smf
    m = d.assign(y=d.mal.astype(int), h=d.halluc.astype(int), typo=d.typo_neighbor.astype(int),
                 z_plaus=(d.plausibility - d.plausibility.mean()) / d.plausibility.std())
    fit = smf.logit("y ~ h + z_plaus + typo + np.log(length) + n_tokens + C(eco)", data=m).fit(disp=0)
    coef = pd.DataFrame({"odds_ratio": np.exp(fit.params), "or_lo": np.exp(fit.conf_int()[0]),
                         "or_hi": np.exp(fit.conf_int()[1]), "p": fit.pvalues}).reset_index(names="term")
    coef.to_csv(TABLES / "u7_adjusted_logit.csv", index=False)
    print(coef.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
