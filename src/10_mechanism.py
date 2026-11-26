"""U1 mechanism: are hallucinated names *targeted*, and how?

(a) malware rate, hallucinated vs matched controls: per ecosystem, pooled Mantel-Haenszel, cutoff sensitivity
(b) conditional on registration: malicious share of claimed hallucinated names vs the ecosystem base rate
    (malicious share of all names newly registered after the cutoff)
(c) dose-response: P(malicious) vs number of 2024 models / mentions emitting the name
(d) campaigns: malicious hits grouped by OSV origin, publication day and registration day
(e) timing: monthly claims, hallucinated vs control, around disclosure milestones (Poisson DiD)
(f) semantic match: does the claimed package's text resemble the prompts that produced the name?
Outputs: tables/u1_*.csv (aggregates only); campaign detail stays in data/derived/.
"""
from __future__ import annotations

import importlib
import io
import json
import re
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from common import DERIVED, REGISTRY, SPRACKLEN, TABLES

st = importlib.import_module("07_stats")
CUTOFFS = ["2024-02-29", "2024-03-31", "2024-04-30"]


def osv_records() -> pd.DataFrame:
    """OSV MAL records for PyPI and npm, excluding withdrawn (retracted) records."""
    mal = pd.read_parquet(REGISTRY / "osv_mal.parquet")
    mal = mal[mal.withdrawn.isna()].copy()
    mal["eco"] = mal.ecosystem.map({"PyPI": "pypi", "npm": "npm"})
    mal["published"] = pd.to_datetime(mal.published, utc=True, format="ISO8601")
    return mal


def osv_post(cutoff) -> set:
    mal = osv_records()
    m = mal[mal.published > cutoff]
    return set(zip(m.eco, m.name))


def osv_post_first(cutoff) -> pd.Series:
    """Earliest post-cutoff MAL publication per (eco, name)."""
    mal = osv_records()
    return mal[mal.published > cutoff].groupby(["eco", "name"]).published.min()


SETS = ("all", "discovery", "c1")


def in_set(df: pd.DataFrame, which: str) -> pd.Series:
    """all: any 2024-era model (pooled); discovery: the five discovery models; c1: the pre-registered
    confirmatory set (an extension model emitted it, no discovery model did)."""
    if which == "discovery":
        return df.n_models_2024d > 0
    if which == "c1":
        return (df.n_models_2024x > 0) & (df.n_models_2024d == 0)
    return df.n_models_2024 > 0


def hallucinated(cutoff, which: str = "all") -> pd.DataFrame:
    cand, _ = st.load_candidates(cutoff)
    h = cand[in_set(cand, which) & (cand.status != "pre_cutoff")].copy()
    post = osv_post(cutoff)
    h["mal_post"] = [(e, n) in post for e, n in zip(h.eco, h.name)]
    return h


def controls(cutoff, which: str = "all", drop: str = "2024") -> pd.DataFrame:
    files = {"discovery": ["controls.parquet"], "c1": ["controls_c1.parquet"],
             "all": ["controls.parquet", "controls_c1.parquet"]}[which]
    parts = [pd.read_parquet(DERIVED / f) for f in files if (DERIVED / f).exists()]
    ctl = pd.concat(parts, ignore_index=True).drop_duplicates(["eco", "name"])
    # a control that a 2024-era model also emitted is itself a hallucination of the studied cohort, not a control.
    # Controls emitted only by 2026 models stay: those models were trained after the cutoff, and could have learned
    # a control name from a package registered later. drop="none" keeps every control (sensitivity analysis).
    if drop != "none":
        names = pd.read_parquet(DERIVED / "names.parquet")
        names = names[names.n_models_2024 > 0]
        ctl = ctl[~ctl.set_index(["eco", "name"]).index.isin(names.set_index(["eco", "name"]).index)].copy()
    ctl["created_ts"] = pd.to_datetime(ctl.created, utc=True, errors="coerce")
    # same exclusions as for hallucinated names (07_stats.status): registered, or tombstoned, before the cutoff
    ctl = ctl[~((ctl.exists | ctl.tombstone.fillna(False).astype(bool)) & ctl.created_ts.notna()
                & (ctl.created_ts <= cutoff))].copy()
    # and, as in 07_stats.status(): a name that no longer exists and whose first malware report predates the
    # cutoff was registered before it
    rec = osv_records()
    first_any = rec.groupby(["eco", "name"]).published.min()
    fa = pd.Series([first_any.get((e, n), pd.NaT) for e, n in zip(ctl.eco, ctl.name)], index=ctl.index)
    gone = ~ctl.exists.fillna(False).astype(bool) & ~ctl.tombstone.fillna(False).astype(bool)
    ctl = ctl[~(gone & fa.notna() & (pd.to_datetime(fa, utc=True) <= cutoff))].copy()
    post = osv_post(cutoff)
    ctl["mal_post"] = [(e, n) in post for e, n in zip(ctl.eco, ctl.name)]
    ctl["claimed"] = ctl.exists | ctl.tombstone | ctl.mal_post
    return ctl


# ---------------------------------------------------------------- (a)
def malware_vs_controls() -> pd.DataFrame:
    """For each name set (all 2024-era models; the five discovery models; confirmatory set C1) and cutoff."""
    rows = []
    sets = [w for w in SETS if w != "c1" or (DERIVED / "controls_c1.parquet").exists()]
    for which, cut in [(w, c) for w in sets for c in CUTOFFS]:
        cutoff = pd.Timestamp(cut, tz="UTC")
        h, c = hallucinated(cutoff, which), controls(cutoff, which)
        tables = []
        for eco in ["npm", "pypi"]:
            a, b = h[h.eco == eco], c[c.eco == eco]
            for outcome, ha, ca in (("malicious", a.mal_post, b.mal_post), ("claimed", a.claimed, b.claimed)):
                ak, an, bk, bn = int(ha.sum()), len(a), int(ca.sum()), len(b)
                orr, p = stats.fisher_exact([[ak, an - ak], [bk, bn - bk]], alternative="greater")
                lo, hi = (np.nan, np.nan)
                if ak and bk:
                    se = np.sqrt(1 / ak - 1 / an + 1 / bk - 1 / bn)
                    rr = (ak / an) / (bk / bn)
                    lo, hi = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)
                rows.append({"set": which, "cutoff": cut, "eco": eco, "outcome": outcome, "halluc_k": ak, "halluc_n": an,
                             "control_k": bk, "control_n": bn, "halluc_rate": ak / an, "control_rate": bk / bn,
                             "risk_ratio": (ak / an) / (bk / bn) if bk else np.inf, "rr_lo": lo, "rr_hi": hi,
                             "fisher_p_one_sided": p})
                if outcome == "malicious":
                    tables.append(np.array([[ak, an - ak], [bk, bn - bk]]))
        mh = StratifiedTable(tables)
        rows.append({"set": which, "cutoff": cut, "eco": "pooled_MH", "outcome": "malicious", "risk_ratio": mh.oddsratio_pooled,
                     "rr_lo": mh.oddsratio_pooled_confint()[0], "rr_hi": mh.oddsratio_pooled_confint()[1],
                     "fisher_p_one_sided": mh.test_null_odds().pvalue})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (b)
def conditional_on_registration(h: pd.DataFrame, cutoff) -> pd.DataFrame:
    """Malicious share among *registered* hallucinated names vs among all names registered after the cutoff."""
    mal = osv_records()
    post = mal[mal.published > cutoff]
    py24 = set((REGISTRY / "pypi_2024.txt").read_text(encoding="utf-8").split("\n"))
    pynow = set((REGISTRY / "pypi_now.txt").read_text(encoding="utf-8").split("\n"))
    py_new_live = len(pynow - py24)
    py_mal = set(post.name[post.eco == "pypi"])
    py_base_den = py_new_live + len(py_mal - pynow)          # live new names + removed malicious ones
    rows = [{"eco": "pypi", "claimed_n": int(h[h.eco == "pypi"].claimed.sum()),
             "claimed_malicious": int((h[h.eco == "pypi"].claimed & h[h.eco == "pypi"].mal_post).sum()),
             "base_new_names": py_base_den, "base_malicious": len(py_mal)}]
    rows.append({"eco": "npm", "claimed_n": int(h[h.eco == "npm"].claimed.sum()),
                 "claimed_malicious": int((h[h.eco == "npm"].claimed & h[h.eco == "npm"].mal_post).sum()),
                 "base_new_names": np.nan, "base_malicious": np.nan,
                 "note": "npm no longer publishes a registry-wide package count; see the matched-control base"})
    df = pd.DataFrame(rows)
    df["claimed_malicious_share"] = df.claimed_malicious / df.claimed_n
    df["base_malicious_share"] = df.base_malicious / df.base_new_names
    df["ratio_vs_registry"] = df.claimed_malicious_share / df.base_malicious_share
    # matched base available for both ecosystems: malicious share among REGISTERED control names
    c = controls(cutoff)
    for i, eco in enumerate(df.eco):
        cc = c[(c.eco == eco) & c.claimed]
        df.loc[i, "control_claimed_n"] = len(cc)
        df.loc[i, "control_claimed_malicious"] = int(cc.mal_post.sum())
        ak, an, bk, bn = int(df.claimed_malicious[i]), int(df.claimed_n[i]), int(cc.mal_post.sum()), len(cc)
        df.loc[i, "control_malicious_share"] = bk / bn if bn else np.nan
        df.loc[i, "ratio_vs_controls"] = (ak / an) / (bk / bn) if bk else np.inf
        df.loc[i, "fisher_p_vs_controls"] = stats.fisher_exact([[ak, an - ak], [bk, bn - bk]], alternative="greater")[1]
    return df


# ---------------------------------------------------------------- (c)
def dose_response(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf
    d = h.assign(y=h.mal_post.astype(int), log_mentions=np.log1p(h.mentions), log_prompts=np.log1p(h.n_prompts))
    d["models_bin"] = pd.cut(d.n_models_2024, [0, 1, 2, 99], labels=["1", "2", "3+"])
    d["mentions_bin"] = pd.cut(d.mentions, [0, 1, 3, 10, 10_000], labels=["1", "2-3", "4-10", ">10"])
    binned = []
    for col in ("models_bin", "mentions_bin"):
        for (eco, b), g in d.groupby(["eco", col], observed=True):
            k, n = int(g.y.sum()), len(g)
            p, lo, hi = st.wilson(k, n)
            binned.append({"eco": eco, "variable": col, "bin": str(b), "n": n, "malicious": k, "rate": p, "ci_lo": lo, "ci_hi": hi})
    fit = smf.logit("y ~ log_mentions + n_models_2024 + C(eco)", data=d).fit(disp=0)
    coef = pd.DataFrame({"coef": fit.params, "odds_ratio": np.exp(fit.params), "p": fit.pvalues,
                         "or_lo": np.exp(fit.conf_int()[0]), "or_hi": np.exp(fit.conf_int()[1])})
    return pd.DataFrame(binned), coef.reset_index(names="term")


# ---------------------------------------------------------------- (d)
def osv_details(hits: set) -> pd.DataFrame:
    rows = []
    for eco, zname in (("pypi", "osv_PyPI_all.zip"), ("npm", "osv_npm_all.zip")):
        want = {n for e, n in hits if e == eco}
        with zipfile.ZipFile(REGISTRY / zname) as z:
            for info in z.infolist():
                if not info.filename.startswith("MAL-"):
                    continue
                j = json.loads(z.read(info))
                names = {a.get("package", {}).get("name") for a in j.get("affected", [])}
                hit = names & want
                if not hit:
                    continue
                ds = j.get("database_specific") or {}
                origins = ds.get("malicious-packages-origins") or []
                rows.append({"eco": eco, "name": sorted(hit)[0], "id": j["id"], "published": j.get("published"),
                             "sources": ",".join(sorted({o.get("source", "?") for o in origins})),
                             "n_versions": sum(len(a.get("versions") or []) for a in j.get("affected", [])),
                             "details": (j.get("details") or "")[:300]})
    return pd.DataFrame(rows)


def campaigns(h: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    hits = set(zip(h.eco[h.mal_post], h.name[h.mal_post]))
    det = osv_details(hits)
    det = det.merge(h[["eco", "name", "created", "mentions", "n_models_2024"]], on=["eco", "name"], how="left")
    det["published_day"] = pd.to_datetime(det.published, utc=True, errors="coerce").dt.date
    det["created_day"] = pd.to_datetime(det.created, utc=True, errors="coerce").dt.date
    det["details_norm"] = det.details.str.replace(r"[\w.\-@/]*" + "", "", regex=False).str[:120]
    det.to_parquet(DERIVED / "u1_malicious_detail.parquet", index=False)      # name-level: not published
    summ = []
    for eco, g in det.groupby("eco"):
        by_pub = g.groupby("published_day").name.nunique()
        by_src = g.assign(src=g.sources.str.split(",")).explode("src").groupby("src").name.nunique()
        summ.append({"eco": eco, "malicious_names": g.name.nunique(), "osv_records": g.id.nunique(),
                     "distinct_publish_days": by_pub.size, "max_names_same_publish_day": int(by_pub.max()),
                     "names_in_same_day_clusters_ge3": int(by_pub[by_pub >= 3].sum()),
                     "sources": json.dumps(by_src.sort_values(ascending=False).to_dict())})
    return pd.DataFrame(summ), det


# ---------------------------------------------------------------- (e)
def timing(h: pd.DataFrame, c: pd.DataFrame) -> pd.DataFrame:
    import statsmodels.formula.api as smf
    first_mal = osv_post_first(st.CUTOFF)

    def monthly(df, group):
        # same dating rule for both arms: first upload if known, else the first post-cutoff OSV report (removed
        # packages); tombstones keep their creation time
        ts = pd.to_datetime(df.created, utc=True, errors="coerce", format="ISO8601")
        mal = pd.Series([first_mal.get((e, n), pd.NaT) for e, n in zip(df.eco, df.name)], index=df.index)
        ts = ts.fillna(pd.to_datetime(mal, utc=True))
        m = ts.dt.tz_convert(None).dt.to_period("M").value_counts().sort_index()
        return pd.DataFrame({"month": m.index.astype(str), "claims": m.values, "group": group, "n_names": len(df)})
    s = pd.concat([monthly(h[h.claimed], "hallucinated"), monthly(c[c.claimed], "control")])
    months = pd.period_range("2024-04", "2026-11", freq="M").astype(str)
    full = pd.MultiIndex.from_product([months, ["hallucinated", "control"]], names=["month", "group"]).to_frame(index=False)
    s = full.merge(s, on=["month", "group"], how="left").fillna({"claims": 0})
    s["n_names"] = s.group.map({"hallucinated": len(h), "control": len(c)})
    s["t"] = s.groupby("group").cumcount()
    s.to_csv(TABLES / "u1_monthly_claims.csv", index=False)
    rows = []
    start = pd.Timestamp("2024-04-01")
    for label, day in st.MILESTONES.items():
        if pd.Timestamp(day) < start + pd.DateOffset(months=2):
            rows.append({"milestone": label, "date": day, "note": "before or too close to series start (2024-04): not estimable"})
            continue
        s["post"] = (pd.PeriodIndex(s.month, freq="M").to_timestamp() >= pd.Timestamp(day)).astype(int)
        s["h"] = (s.group == "hallucinated").astype(int)
        try:
            fit = smf.glm("claims ~ h * post + t", data=s, family=__import__("statsmodels.api").api.families.Poisson(),
                          offset=np.log(s.n_names)).fit()
            rows.append({"milestone": label, "date": day, "did_irr": np.exp(fit.params["h:post"]),
                         "irr_lo": np.exp(fit.conf_int().loc["h:post", 0]), "irr_hi": np.exp(fit.conf_int().loc["h:post", 1]),
                         "p": fit.pvalues["h:post"]})
        except Exception as ex:  # noqa: BLE001
            rows.append({"milestone": label, "date": day, "error": repr(ex)[:200]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- (f)
_TOK = re.compile(r"[a-z][a-z0-9]{2,}")
_STOP = set("the and for with that this from your you are use using code how can not what when which into have has "
            "will would should could file data value values function method class return python javascript package "
            "packages module library npm pypi install example get set new make want need way does".split())


def toks(s: str) -> set:
    return {t for t in _TOK.findall((s or "").lower()) if t not in _STOP}


def semantic(h: pd.DataFrame, seed: int = 1122) -> pd.DataFrame:
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")
    prompts = {}
    for lang in ("Python", "Javascript"):
        for sset in ("SO_AT", "SO_LY", "LLM_AT", "LLM_LY"):
            lines = [l for l in (SPRACKLEN / "Data" / lang / f"{sset}.json").read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, l in enumerate(lines):
                o = json.loads(l)
                prompts[(lang, sset, i)] = str(o if isinstance(o, str) else next(iter(o.values())))[:2000]
    live = h[h.claimed & (h.status == "claimed_live")].copy()
    live["text"] = (live.summary.fillna("") + " " + live.description_head.fillna("")).str[:3000]
    live = live[live.text.str.len() > 20]
    m = mentions[mentions.era == 2024].merge(live[["eco", "name"]], on=["eco", "name"])
    ptxt = m.groupby(["eco", "name"]).apply(lambda g: " ".join(prompts.get((r.lang, r.set, r.idx), "") for r in g.itertuples()))
    live = live.set_index(["eco", "name"])

    def name_toks(n: str) -> set:          # the package name's own words appear on both sides by construction: remove
        parts = re.split(r"[-_./@]+|(?<=[a-z])(?=[A-Z])", n)
        return {p.lower() for p in parts if p} | {n.lower()}
    pairs = [(toks(ptxt.get(k, "")) - name_toks(k[1]), toks(live.loc[k, "text"]) - name_toks(k[1]), k[0])
             for k in live.index if k in ptxt.index]
    def jac(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0
    obs = [jac(a, b) for a, b, _ in pairs]
    rng = np.random.default_rng(seed)
    shuf = []                               # shuffle within ecosystem, so Python prompts meet PyPI text only
    for eco in ("npm", "pypi"):
        idx = [i for i, p in enumerate(pairs) if p[2] == eco]
        perm = rng.permutation(idx)
        shuf += [jac(pairs[i][0], pairs[j][1]) for i, j in zip(idx, perm) if i != j]
    w = stats.mannwhitneyu(obs, shuf, alternative="greater")
    return pd.DataFrame([{"n_pairs": len(obs), "jaccard_median": float(np.median(obs)), "jaccard_mean": float(np.mean(obs)),
                          "shuffled_median": float(np.median(shuf)), "shuffled_mean": float(np.mean(shuf)),
                          "mannwhitney_p": w.pvalue}])


def main() -> None:
    a = malware_vs_controls()
    a.to_csv(TABLES / "u1a_malware_vs_controls.csv", index=False)
    print(a[a.cutoff == "2024-03-31"].to_string(index=False))
    cutoff = st.CUTOFF
    h, c = hallucinated(cutoff), controls(cutoff)
    b = conditional_on_registration(h, cutoff)
    b.to_csv(TABLES / "u1b_conditional_on_registration.csv", index=False)
    print(b.to_string(index=False))
    binned, coef = dose_response(h)
    binned.to_csv(TABLES / "u1c_dose_response_bins.csv", index=False)
    coef.to_csv(TABLES / "u1c_dose_response_logit.csv", index=False)
    print(binned.to_string(index=False)); print(coef.to_string(index=False))
    summ, _ = campaigns(h)
    summ.to_csv(TABLES / "u1d_campaigns.csv", index=False)
    print(summ.to_string(index=False))
    t = timing(h, c)
    t.to_csv(TABLES / "u1e_timing_did.csv", index=False)
    print(t.to_string(index=False))
    s = semantic(h)
    s.to_csv(TABLES / "u1f_semantic_match.csv", index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
