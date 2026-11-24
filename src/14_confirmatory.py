"""Confirmatory tests H1-H6, exactly as specified in docs/analysis_plan.md (committed 2026-11-24 00:33 IST).

H1 (primary)  C1 names vs C1 controls, post-cutoff OSV MAL record. Pooled one-sided Fisher exact;
              CMH stratified by ecosystem as secondary. Supported if OR > 1 and p < 0.05.
H2            Same, outcome = registered at all (claimed).
H3            All 2024-cohort names: logit malicious ~ log1p(mentions) + n_models_2024 + C(eco); supported
              if the n_models coefficient > 0 with p < 0.05.
H4            Of still-unregistered 2024-cohort names, share emitted by >= 1 2026 local model; prediction >= 1%.
H5            Names that carry or carried malware (post-cutoff) still emitted by 2026 models: counts per model.
H6            2026-only names (never emitted by any 2024-era model) have a lower post-cutoff malicious rate
              than 2024-cohort names. One-sided Fisher exact.
Completeness of every model's generations is printed first; a test whose inputs are incomplete is marked.
Output: tables/confirmatory.csv, tables/h5_live_risk_by_model.csv (aggregates only).
"""
from __future__ import annotations

import importlib
import json

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats.contingency import odds_ratio
from statsmodels.stats.contingency_tables import StratifiedTable

from common import ALL_MODELS, DERIVED, EXTENSION, GEN, MODELS, TABLES

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")
N_PROMPTS = 3000


def completeness() -> dict:
    out = {}
    for m in ALL_MODELS:
        p = GEN / (m.replace(":", "__").replace("/", "_") + ".jsonl")
        out[m] = sum(1 for _ in open(p, encoding="utf-8")) if p.exists() else 0
    return out


def two_by_two(ak, an, bk, bn, alternative="greater") -> dict:
    t = np.array([[ak, an - ak], [bk, bn - bk]])
    res = odds_ratio(t, kind="conditional")
    ci = res.confidence_interval(0.95)
    return {"a_k": ak, "a_n": an, "b_k": bk, "b_n": bn, "a_rate": ak / an if an else np.nan,
            "b_rate": bk / bn if bn else np.nan, "odds_ratio": res.statistic, "or_lo": ci.low, "or_hi": ci.high,
            "p_one_sided": stats.fisher_exact(t, alternative=alternative)[1]}


def cmh(h: pd.DataFrame, c: pd.DataFrame, col_h: str, col_c: str) -> dict:
    tabs = []
    for eco in ("npm", "pypi"):
        a, b = h[h.eco == eco], c[c.eco == eco]
        if len(a) and len(b):
            tabs.append(np.array([[a[col_h].sum(), len(a) - a[col_h].sum()], [b[col_c].sum(), len(b) - b[col_c].sum()]], float))
    s = StratifiedTable(tabs)
    lo, hi = s.oddsratio_pooled_confint()
    return {"cmh_or": s.oddsratio_pooled, "cmh_lo": lo, "cmh_hi": hi, "cmh_p_two_sided": s.test_null_odds().pvalue}


def main() -> None:
    comp = completeness()
    print("generations per model:", json.dumps(comp, indent=1))
    ext_done = all(comp[m] >= N_PROMPTS for m in EXTENSION)
    loc26_done = all(comp[m] >= N_PROMPTS for m, v in MODELS.items() if v["era"] == 2026)
    cutoff = st.CUTOFF
    rows = []

    # H1 / H2
    h, c = mech.hallucinated(cutoff, "c1"), mech.controls(cutoff, "c1") if (DERIVED / "controls_c1.parquet").exists() else None
    if c is None or h.empty:
        rows.append({"hypothesis": "H1", "note": "C1 or its controls not built yet (run 06_controls.py --set c1)"})
    else:
        for hyp, col in (("H1", "mal_post"), ("H2", "claimed")):
            r = two_by_two(int(h[col].sum()), len(h), int(c[col].sum()), len(c))
            r.update(cmh(h, c, col, col))
            r["supported"] = bool(r["odds_ratio"] > 1 and r["p_one_sided"] < 0.05)
            rows.append({"hypothesis": hyp, "complete_inputs": ext_done, **r})
            for eco in ("npm", "pypi"):
                a, b = h[h.eco == eco], c[c.eco == eco]
                if len(a) and len(b):
                    rows.append({"hypothesis": f"{hyp}_{eco}", "complete_inputs": ext_done,
                                 **two_by_two(int(a[col].sum()), len(a), int(b[col].sum()), len(b))})

    # H3
    import statsmodels.formula.api as smf
    a = mech.hallucinated(cutoff, "all")
    d = a.assign(y=a.mal_post.astype(int))
    fit = smf.logit("y ~ np.log1p(mentions) + n_models_2024 + C(eco)", data=d).fit(disp=0)
    b, p2 = fit.params["n_models_2024"], fit.pvalues["n_models_2024"]
    rows.append({"hypothesis": "H3", "complete_inputs": ext_done, "coef_n_models": b, "odds_ratio": np.exp(b),
                 "or_lo": np.exp(fit.conf_int().loc["n_models_2024", 0]), "or_hi": np.exp(fit.conf_int().loc["n_models_2024", 1]),
                 "p_two_sided": p2, "p_one_sided": p2 / 2 if b > 0 else 1 - p2 / 2, "a_n": len(d), "a_k": int(d.y.sum()),
                 "supported": bool(b > 0 and p2 < 0.05)})

    # H4
    un = a[a.status == "unregistered"]
    k, n = int((un.n_models_2026 > 0).sum()), len(un)
    p, lo, hi = st.wilson(k, n)
    rows.append({"hypothesis": "H4", "complete_inputs": loc26_done and ext_done, "a_k": k, "a_n": n, "a_rate": p,
                 "rate_lo": lo, "rate_hi": hi, "supported": bool(p >= 0.01), "ci_excludes_below_1pct": bool(lo >= 0.01)})

    # H5: per-model counts of malicious names (post-cutoff MAL) still emitted
    mal = a[a.mal_post][["eco", "name"]]
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")
    m26 = mentions[mentions.era == 2026].merge(mal, on=["eco", "name"])
    h5 = (m26.groupby(["cohort", "model", "eco"]).name.nunique().rename("malicious_names_emitted").reset_index())
    h5.to_csv(TABLES / "h5_live_risk_by_model.csv", index=False)
    rows.append({"hypothesis": "H5", "complete_inputs": loc26_done, "a_k": int(m26[["eco", "name"]].drop_duplicates().shape[0]),
                 "note": "distinct malicious 2024-cohort names emitted by >= 1 2026 model; per model in h5_live_risk_by_model.csv"})

    # H6: 2026-only names vs 2024-cohort names
    cand, _ = st.load_candidates(cutoff)
    only26 = cand[(cand.n_models_2024 == 0) & ((cand.n_models_2026 + cand.n_models_2026h) > 0) & (cand.status != "pre_cutoff")].copy()
    post = mech.osv_post(cutoff)
    only26["mal_post"] = [(e, nm) in post for e, nm in zip(only26.eco, only26.name)]
    r = two_by_two(int(only26.mal_post.sum()), len(only26), int(a.mal_post.sum()), len(a), alternative="less")
    r["supported"] = bool(r["odds_ratio"] < 1 and r["p_one_sided"] < 0.05)
    rows.append({"hypothesis": "H6", "complete_inputs": loc26_done and ext_done,
                 "note": "a = 2026-only names, b = 2024-cohort names; one-sided (a < b)", **r})

    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "confirmatory.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(5).to_string(index=False))
        print(h5.to_string(index=False))


if __name__ == "__main__":
    main()
