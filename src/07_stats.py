"""Statistics for RQ1-RQ4. Writes aggregate tables to tables/ (safe to publish: no unregistered names).

Status of a candidate name (absent from the early-2024 snapshot):
  pre_cutoff       exists, first upload <= CUTOFF          -> snapshot gap, excluded
  claimed_live     exists, first upload  > CUTOFF
  claimed_nofiles  PyPI project exists but has no files (name held, date unknown)
  claimed_removed  gone now, but OSV MAL record or npm unpublish tombstone dated > CUTOFF
  unregistered     no trace of registration
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from common import DERIVED, MODELS, REGISTRY, TABLES

CUTOFF = pd.Timestamp("2024-03-31", tz="UTC")
MILESTONES = {
    "Lanyado huggingface-cli report": "2024-03-28",
    "Spracklen et al. arXiv": "2024-06-12",
    "'slopsquatting' coined": "2025-04-12",
    "USENIX Security 2025": "2025-08-13",
}


def status(r, cutoff) -> str:
    created = pd.to_datetime(r.get("created"), utc=True, errors="coerce")
    malpub = pd.to_datetime(r.get("mal_first_published"), utc=True, errors="coerce")
    if r.get("exists") is True or r.get("exists") == 1:
        if pd.isna(created):
            return "claimed_nofiles"
        return "claimed_live" if created > cutoff else "pre_cutoff"
    if r.get("tombstone") and not pd.isna(created):
        return "claimed_removed" if created > cutoff else "pre_cutoff"
    if r.get("osv_mal"):
        return "claimed_removed" if (pd.isna(malpub) or malpub > cutoff) else "pre_cutoff"
    return "unregistered"


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return (np.nan, np.nan, np.nan)
    lo, hi = proportion_confint(k, n, method="wilson")
    return (k / n, lo, hi)


def rate_table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by) if by else [((), df)]:
        n = len(g)
        k = int(g.claimed.sum())
        p, lo, hi = wilson(k, n)
        rows.append({**dict(zip(by, key if isinstance(key, tuple) else (key,))), "eligible": n, "claimed": k,
                     "rate": p, "ci_lo": lo, "ci_hi": hi, "malicious": int(g.malicious.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default=str(CUTOFF.date()))
    a = ap.parse_args()
    cutoff = pd.Timestamp(a.cutoff, tz="UTC")
    names = pd.read_parquet(DERIVED / "names.parquet")
    enr = pd.read_parquet(DERIVED / "enriched.parquet")
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")

    cand = names[~names.in_2024].merge(enr, on=["eco", "name"], how="left")
    cand["status"] = [status(r, cutoff) for r in cand.to_dict("records")]
    cand["claimed"] = cand.status.str.startswith("claimed")
    cand["malicious"] = cand.osv_mal.fillna(False).astype(bool)
    elig = cand[cand.status != "pre_cutoff"].copy()
    h24 = elig[elig.n_models_2024 > 0]

    out = {"cutoff": str(cutoff.date()), "status_counts_2024_cohort":
           cand[cand.n_models_2024 > 0].groupby(["eco", "status"]).size().unstack(fill_value=0).to_dict("index")}

    # RQ1: claim rate of names hallucinated by the 2024-era cohort
    t1 = rate_table(h24, ["eco"])
    t1_dot = rate_table(h24, ["eco", "only_dotted"])
    t1.to_csv(TABLES / "rq1_claim_rate.csv", index=False)
    t1_dot.to_csv(TABLES / "rq1_claim_rate_by_dotted.csv", index=False)

    # per-model hallucination share and claim rate
    m24 = mentions[mentions.era == 2024]
    hall = m24.merge(names[["eco", "name", "in_2024"]], on=["eco", "name"])
    per_model = hall.groupby(["model", "eco"]).agg(mentions=("name", "size"),
                                                   hallucinated_share=("in_2024", lambda s: 1 - s.mean())).reset_index()
    claimed_names = set(zip(h24.eco[h24.claimed], h24.name[h24.claimed]))
    per_model["claimed_unique"] = [
        len(set(zip(g.eco, g.name)) & claimed_names)
        for g in (hall[(hall.model == m) & (hall.eco == e)] for m, e in zip(per_model.model, per_model.eco))]
    per_model.to_csv(TABLES / "per_model.csv", index=False)

    # RQ2: hallucinated vs matched controls
    ctl_path = DERIVED / "controls.parquet"
    if ctl_path.exists():
        ctl = pd.read_parquet(ctl_path)
        ctl["created_ts"] = pd.to_datetime(ctl.created, utc=True, errors="coerce")
        ctl = ctl[~(ctl.exists & ctl.created_ts.notna() & (ctl.created_ts <= cutoff))]
        ctl["claimed"] = ctl.exists | ctl.tombstone
        rows = []
        for eco in sorted(set(h24.eco)):
            a_k, a_n = int(h24[h24.eco == eco].claimed.sum()), int((h24.eco == eco).sum())
            c = ctl[ctl.eco == eco]
            b_k, b_n = int(c.claimed.sum()), len(c)
            orr, p = stats.fisher_exact([[a_k, a_n - a_k], [b_k, b_n - b_k]]) if a_n and b_n else (np.nan, np.nan)
            rows.append({"eco": eco, "halluc_claimed": a_k, "halluc_n": a_n, "control_claimed": b_k, "control_n": b_n,
                         "halluc_rate": a_k / a_n if a_n else np.nan, "control_rate": b_k / b_n if b_n else np.nan,
                         "risk_ratio": (a_k / a_n) / (b_k / b_n) if a_n and b_n and b_k else np.nan,
                         "odds_ratio": orr, "fisher_p": p})
        pd.DataFrame(rows).to_csv(TABLES / "rq2_vs_controls.csv", index=False)

    # Timing of claims relative to milestones (aggregate only)
    cl = h24[h24.claimed].copy()
    # event date: first upload if known, else the OSV malware report date (removed packages)
    cl["created_ts"] = pd.to_datetime(cl.created, utc=True, errors="coerce").fillna(
        pd.to_datetime(cl.mal_first_published, utc=True, errors="coerce"))
    cl["created_ts"] = cl.created_ts.dt.tz_convert(None)
    cl["quarter"] = cl.created_ts.dt.to_period("Q").astype(str)
    cl.groupby(["eco", "quarter"]).size().rename("claims").reset_index().to_csv(TABLES / "claims_by_quarter.csv", index=False)
    out["claims_after_milestone"] = {k: int((cl.created_ts > pd.Timestamp(v)).sum()) for k, v in MILESTONES.items()}
    out["claims_with_known_date"] = int(cl.created_ts.notna().sum())

    # RQ3: persistence of 2024-era hallucinations in the 2026 cohort
    pers = rate_table(h24.assign(claimed=h24.n_models_2026 > 0), ["eco"]).rename(
        columns={"claimed": "also_emitted_by_2026", "rate": "persistence"})
    pers["cohort"] = "2026-local"
    if "n_models_2026h" in h24:
        pers_h = rate_table(h24.assign(claimed=h24.n_models_2026h > 0), ["eco"]).rename(
            columns={"claimed": "also_emitted_by_2026", "rate": "persistence"})
        pers_h["cohort"] = "2026-hosted"
        pers = pd.concat([pers, pers_h], ignore_index=True)
        out["persistent_and_unregistered_hosted"] = h24[(h24.n_models_2026h > 0) & (h24.status == "unregistered")].groupby("eco").size().to_dict()
    pers.to_csv(TABLES / "rq3_persistence.csv", index=False)
    out["persistent_and_unregistered"] = h24[(h24.n_models_2026 > 0) & (h24.status == "unregistered")].groupby("eco").size().to_dict()

    # RQ4: predictors of being claimed (logistic regression)
    try:
        import statsmodels.formula.api as smf
        d = h24.assign(y=h24.claimed.astype(int), log_mentions=np.log1p(h24.mentions), length=h24.name.str.len(),
                       n_tokens=h24.name.str.count("-") + 1, dotted=h24.only_dotted.astype(int))
        if d.y.sum() >= 10:
            fit = smf.logit("y ~ log_mentions + n_models + length + n_tokens + dotted + C(eco)", data=d).fit(disp=0)
            pd.DataFrame({"coef": fit.params, "or": np.exp(fit.params), "p": fit.pvalues,
                          "ci_lo": np.exp(fit.conf_int()[0]), "ci_hi": np.exp(fit.conf_int()[1])}).to_csv(TABLES / "rq4_logit.csv")
            out["rq4_n"], out["rq4_events"] = int(len(d)), int(d.y.sum())
        else:
            out["rq4"] = f"skipped: only {int(d.y.sum())} events"
    except Exception as ex:  # noqa: BLE001
        out["rq4_error"] = repr(ex)

    out["rq1"] = t1.to_dict("records")
    (TABLES / "summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str)[:3000])


if __name__ == "__main__":
    main()
