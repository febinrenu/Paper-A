"""Every number quoted in the manuscript, as LaTeX macros generated from tables/ (no hand-typed results).

Output: paper/numbers.tex. A value whose table is missing or not yet computed renders as a bold red "??" so it
cannot slip into a submitted draft unnoticed. Rerun after the final analysis: python src/15_numbers.py
"""
from __future__ import annotations

import json
import math

import pandas as pd

from common import ROOT, TABLES

OUT = ROOT / "paper" / "numbers.tex"
MISSING = r"\textcolor{red}{\textbf{??}}"
macros: dict[str, str] = {}


def name_ok(k: str) -> str:
    assert k.isalpha(), k
    return k


def put(k: str, v) -> None:
    macros[name_ok(k)] = v if isinstance(v, str) else MISSING


def num(v, d=1) -> str | None:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return None
    if d == 0:
        return f"{int(round(v)):,}".replace(",", "{,}")
    return f"{v:.{d}f}"


def pct(v, d=1) -> str | None:
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.{d}f}\\%"


def pval(p) -> str | None:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return None
    if p == 0 or p < 1e-15:
        return r"$p<10^{-15}$"
    if p < 0.001:
        e = math.floor(math.log10(p))
        return rf"$p={p / 10 ** e:.1f}\times10^{{{e}}}$"
    return f"$p={p:.3f}$" if p < 0.1 else f"$p={p:.2f}$"


def read(name: str) -> pd.DataFrame | None:
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


ECO = {"npm": "Npm", "pypi": "Pypi"}
SETW = {"all": "All", "discovery": "Disc", "c1": "Conf"}
CUT = {"2024-03-31": "", "2024-02-29": "Feb", "2024-04-30": "Apr"}


def malware() -> None:
    a = read("u1a_malware_vs_controls.csv")
    if a is None:
        return
    if "set" not in a:
        a["set"] = "all"
    for (s, cut, eco, out), r in a.set_index(["set", "cutoff", "eco", "outcome"]).iterrows():
        if s not in SETW or cut not in CUT:
            continue
        pre = SETW[s] + CUT[cut]
        if eco == "pooled_MH":
            put(f"{pre}PooledOR", num(r.risk_ratio))
            put(f"{pre}PooledLo", num(r.rr_lo))
            put(f"{pre}PooledHi", num(r.rr_hi))
            put(f"{pre}PooledP", pval(r.fisher_p_one_sided))
            continue
        o = "Mal" if out == "malicious" else "Claim"
        e = ECO[eco]
        put(f"{pre}{o}{e}K", num(r.halluc_k, 0))
        put(f"{pre}{o}{e}N", num(r.halluc_n, 0))
        put(f"{pre}{o}{e}CtlK", num(r.control_k, 0))
        put(f"{pre}{o}{e}CtlN", num(r.control_n, 0))
        put(f"{pre}{o}{e}Rate", pct(r.halluc_rate, 2))
        put(f"{pre}{o}{e}CtlRate", pct(r.control_rate, 3))
        put(f"{pre}{o}{e}RR", num(r.risk_ratio))
        put(f"{pre}{o}{e}Lo", num(r.rr_lo))
        put(f"{pre}{o}{e}Hi", num(r.rr_hi))
        put(f"{pre}{o}{e}P", pval(r.fisher_p_one_sided))


def conditional() -> None:
    b = read("u1b_conditional_on_registration.csv")
    if b is None:
        return
    for _, r in b.iterrows():
        e = ECO[r.eco]
        put(f"Cond{e}N", num(r.claimed_n, 0))
        put(f"Cond{e}K", num(r.claimed_malicious, 0))
        put(f"Cond{e}Share", pct(r.claimed_malicious_share))
        put(f"Cond{e}CtlShare", pct(r.control_malicious_share))
        put(f"Cond{e}Ratio", num(r.ratio_vs_controls))
        put(f"Cond{e}P", pval(r.fisher_p_vs_controls))
        if r.eco == "pypi":
            put("CondPypiBase", pct(r.base_malicious_share, 2))
            put("CondPypiBaseRatio", num(r.ratio_vs_registry))
            put("CondPypiBaseN", num(r.base_new_names, 0))


def dose() -> None:
    d = read("u1c_dose_response_bins.csv")
    if d is not None:
        for _, r in d.iterrows():
            tag = {"1": "One", "2-3": "Two", "4-10": "Four", ">10": "Ten", "2": "Two", "3+": "Three"}.get(str(r.bin))
            v = "Mentions" if r.variable == "mentions_bin" else "Models"
            if tag:
                put(f"Dose{ECO[r.eco]}{v}{tag}", pct(r.rate))
                put(f"Dose{ECO[r.eco]}{v}{tag}N", num(r.n, 0))
    c = read("u1c_dose_response_logit.csv")
    if c is not None:
        col = "term" if "term" in c else c.columns[0]
        for _, r in c.iterrows():
            t = str(r[col])
            key = "Mentions" if "mention" in t else "Models" if "n_models" in t else None
            if key and "odds_ratio" in c:
                put(f"DoseLogit{key}OR", num(r.odds_ratio, 2))
                put(f"DoseLogit{key}P", pval(r.p))


def campaigns_timing_semantic() -> None:
    c = read("u1d_campaigns.csv")
    if c is not None:
        for _, r in c.iterrows():
            e = ECO[r.eco]
            put(f"Camp{e}Names", num(r.malicious_names, 0))
            put(f"Camp{e}Days", num(r.distinct_publish_days, 0))
            put(f"Camp{e}MaxDay", num(r.max_names_same_publish_day, 0))
            put(f"Camp{e}Clustered", num(r.names_in_same_day_clusters_ge3, 0))
    t = read("u1e_timing_did.csv")
    if t is not None:
        for _, r in t.iterrows():
            k = {"Spracklen et al. arXiv": "Arxiv", "'slopsquatting' coined": "Coined", "USENIX Security 2025": "Usenix"}.get(r.milestone)
            if k:
                put(f"Did{k}IRR", num(r.did_irr, 2))
                put(f"Did{k}Lo", num(r.irr_lo, 2))
                put(f"Did{k}Hi", num(r.irr_hi, 2))
                put(f"Did{k}P", pval(r.p))
    s = read("u1f_semantic_match.csv")
    if s is not None:
        r = s.iloc[0]
        put("SemN", num(r.n_pairs, 0))
        put("SemMean", num(r.jaccard_mean, 3))
        put("SemShufMean", num(r.shuffled_mean, 3))
        put("SemP", pval(r.mannwhitney_p))


def confounds() -> None:
    u = read("u7_stratified_or.csv")
    if u is not None:
        keys = {"crude (eco strata)": "Crude", "eco x plausibility quintile": "Plaus",
                "no typo neighbour, eco x plausibility quintile": "NoTypo",
                "typo neighbour only, eco x plausibility quintile": "TypoOnly",
                "PyPI, not one edit from a top-5k package, plausibility quintile": "NoTop",
                "npm, not one edit from a top npm package, plausibility quintile": "NoTopNpm",
                "both, not one edit from a top package, eco x plausibility quintile": "NoTopBoth"}
        for _, r in u.iterrows():
            k = keys.get(r.analysis)
            if k:
                put(f"Conf{k}OR", num(r.or_mh))
                put(f"Conf{k}Lo", num(r.or_lo))
                put(f"Conf{k}Hi", num(r.or_hi))
                put(f"Conf{k}P", pval(r.p))
    a = read("u7_adjusted_logit.csv")
    if a is not None:
        a = a.set_index("term")
        for t, k in (("h", "Adj"), ("z_plaus", "AdjPlaus")):
            if t in a.index:
                put(f"{k}OR", num(a.loc[t, "odds_ratio"], 2))
                put(f"{k}Lo", num(a.loc[t, "or_lo"], 2))
                put(f"{k}Hi", num(a.loc[t, "or_hi"], 2))
                put(f"{k}P", pval(a.loc[t, "p"]))
    b = read("u7_balance.csv")
    if b is not None and "std_diff" in b:
        for _, r in b[b.variable == "plausibility"].iterrows():
            put(f"BalPlausSMD{ECO[r.eco]}", num(r.std_diff, 2))
        for _, r in b[b.variable == "typo_neighbor"].iterrows():
            put(f"BalTypo{ECO[r.eco]}H", pct(r.halluc_mean, 0)); put(f"BalTypo{ECO[r.eco]}C", pct(r.control_mean, 0))
        for _, r in b[b.variable == "n_tokens"].iterrows():
            put(f"BalTok{ECO[r.eco]}H", num(r.halluc_mean, 2)); put(f"BalTok{ECO[r.eco]}C", num(r.control_mean, 2))


def confirmatory() -> None:
    c = read("confirmatory.csv")
    if c is None:
        return
    c = c.set_index("hypothesis")
    w = {"H1": "HOne", "H2": "HTwo", "H3": "HThree", "H4": "HFour", "H5": "HFive", "H6": "HSix",
         "H1_npm": "HOneNpm", "H1_pypi": "HOnePypi", "H2_npm": "HTwoNpm", "H2_pypi": "HTwoPypi"}
    for h, k in w.items():
        if h not in c.index:
            continue
        r = c.loc[h]
        for col, suf, f in (("odds_ratio", "OR", lambda v: num(v, 2)), ("or_lo", "Lo", lambda v: num(v, 2)),
                            ("or_hi", "Hi", lambda v: num(v, 2)), ("p_one_sided", "P", pval),
                            ("a_k", "K", lambda v: num(v, 0)), ("a_n", "N", lambda v: num(v, 0)),
                            ("b_k", "CtlK", lambda v: num(v, 0)), ("b_n", "CtlN", lambda v: num(v, 0)),
                            ("a_rate", "Rate", lambda v: pct(v, 2)), ("b_rate", "CtlRate", lambda v: pct(v, 3)),
                            ("rate_lo", "RateLo", lambda v: pct(v, 1)), ("rate_hi", "RateHi", lambda v: pct(v, 1)),
                            ("cmh_or", "CMH", lambda v: num(v, 2)), ("cmh_lo", "CMHLo", lambda v: num(v, 2)),
                            ("cmh_hi", "CMHHi", lambda v: num(v, 2))):
            if col in c.columns and pd.notna(r.get(col)):
                put(k + suf, f(float(r[col])))
        if "supported" in c.columns and pd.notna(r.get("supported")):
            put(k + "Verdict", "supported" if str(r.supported) == "True" else "not supported")
    if "H4" in c.index:
        put("HFourRate", pct(c.loc["H4", "a_rate"], 1))
    h5 = read("h5_live_risk_by_model.csv")
    if h5 is not None:
        put("HFiveModels", num(h5.model.nunique(), 0))
        put("HFiveMaxModel", h5.groupby("model").malicious_names_emitted.sum().idxmax().replace("_", r"\_"))
        put("HFiveMax", num(h5.groupby("model").malicious_names_emitted.sum().max(), 0))
        hh = h5[h5.cohort == "2026-hosted"].groupby("model").malicious_names_emitted.sum()
        if len(hh):
            put("HostedModels", num(len(hh), 0))
            put("HostedMalMin", num(hh.min(), 0)); put("HostedMalMax", num(hh.max(), 0))
    s = TABLES / "summary.json"
    if s.exists():
        j = json.loads(s.read_text())
        for key, k in (("persistent_and_unregistered_hosted", "Hosted"), ("persistent_and_unregistered", "Local")):
            for e, v in (j.get(key) or {}).items():
                put(f"PersistUnreg{k}{ECO[e]}", num(v, 0))


def exposure_claimants_defense() -> None:
    e = read("u2_exposure_summary.csv")
    if e is not None:
        g = {"hallucinated_malicious": "Mal", "hallucinated_other": "Other", "control_claimed": "Ctl"}
        for _, r in e.iterrows():
            k = g.get(r.subgroup)
            if k:
                put(f"Exp{ECO[r.eco]}{k}N", num(r.n, 0))
                put(f"Exp{ECO[r.eco]}{k}Median", num(r.downloads_median, 0))
                put(f"Exp{ECO[r.eco]}{k}Total", num(r.downloads_total, 0))
                put(f"Exp{ECO[r.eco]}{k}Dependents", num(r.dependents_total, 0))
                if pd.notna(r.get("takedown_days_median")):
                    put(f"Exp{ECO[r.eco]}{k}TakedownDays", num(r.takedown_days_median, 1))
    c = read("u6_claimants.csv")
    if c is not None:
        for _, r in c.iterrows():
            k = "".join(w.capitalize() for w in str(r["class"]).split("_"))
            put(f"Cls{ECO[r.eco]}{k}", num(r.n, 0))
        tot = c.groupby("eco").n.sum()
        for e, v in tot.items():
            put(f"Cls{ECO[e]}Total", num(v, 0))
    d = read("rq5_defense_summary.csv")
    if d is not None:
        for _, r in d[d.policy.isin(["mentions", "logit_cv"])].iterrows():
            k = "Freq" if r.policy == "mentions" else "Model"
            put(f"Def{ECO[r.eco]}{k}KHalfMal", num(r.K_for_50pct_malicious, 0))
            put(f"Def{ECO[r.eco]}{k}KHalfMalPct", pct(r.K_for_50pct_malicious / r.candidates, 0))
            put(f"Def{ECO[r.eco]}Candidates", num(r.candidates, 0))
    o = read("rq5_out_of_sample.csv")
    if o is not None:
        for _, r in o.iterrows():
            e = {"npm": "Npm", "pypi": "Pypi", "pooled": ""}[r.eco]
            k = {"malicious": "", "claimed": "Claimed", "all_names": "Names"}[r.outcome]
            put(f"DefOos{e}{k}Recall", pct(r.recall, 0))
            put(f"DefOos{e}{k}K", num(r.on_discovery_watchlist, 0))
            put(f"DefOos{e}{k}N", num(r.n, 0))
    pb = read("rq5_probe_budget.csv")
    if pb is not None:
        for eco in ("npm", "pypi"):
            x = pb[pb.eco == eco]
            if len(x):
                full = x[(x.models == x.models.max()) & (x.prompts_per_model == x.prompts_per_model.max())].iloc[0]
                put(f"Probe{ECO[eco]}FullHours", num(full.gpu_hours, 0))
                put("ProbeModels", num(x.models.max(), 0))
                for m, w in ((1, "One"), (3, "Three")):       # share of all later-malicious names, full prompts
                    y = x[(x.models == m) & (x.prompts_per_model == x.prompts_per_model.max())]
                    if len(y):
                        put(f"Probe{ECO[eco]}{w}ModelRecall", pct(y.malicious_recall.iloc[0], 0))
                small = x[(x.models == 3) & (x.prompts_per_model == 1000)]
                if len(small):
                    put(f"Probe{ECO[eco]}ThreeKRecall", pct(small.malicious_recall.iloc[0], 0))
                    put(f"Probe{ECO[eco]}ThreeKHours", num(small.gpu_hours.iloc[0], 1))


def detection_manifests() -> None:
    d = read("u2b_detection.csv")
    if d is not None:
        lab = {"report_days": "Report", "window_days": "Window", "window_days_affected_only": "WindowAff"}
        for _, r in d[d.measure.isin(list(lab))].iterrows():
            if not r.n:
                continue
            k = f"Det{ECO[r.eco]}{lab[r.measure]}"
            put(k + "Median", num(r["median"], 1)); put(k + "PTwentyFive", num(r.p25, 1))
            put(k + "PSeventyFive", num(r.p75, 1)); put(k + "N", num(r.n, 0)); put(k + "Of", num(r["of"], 0))
            put(k + "GtSeven", pct(r.share_gt_7d, 0)); put(k + "GtThirty", pct(r.share_gt_30d, 0))
            put(k + "LeOne", pct(r.share_le_1d, 0))
        for _, r in d[d.measure == "report_vs_removal"].iterrows():
            put(f"Det{ECO[r.eco]}CoincideShare", pct(r.share_le_10min, 0))
            put(f"Det{ECO[r.eco]}BatchMax", num(r.max_names_same_report_hour, 0))
        m = d[d.measure == "methods"]
        for _, r in m.iterrows():
            for col, k in (("affected_versions", "AffectedVersions"), ("security_placeholder", "Placeholder")):
                if col in r and pd.notna(r[col]):
                    put(f"Det{ECO[r.eco]}{k}", num(r[col], 0))
    w = read("u9_watchlist.csv")
    if w is not None:
        for _, r in w.iterrows():
            put(f"Watch{ECO[r.eco]}", num(r.names, 0))
        put("WatchTotal", num(w.names.sum(), 0))
    g = read("u2c_manifests.csv")
    if g is not None:
        for _, r in g.iterrows():
            k = f"Man{ECO[r.eco]}"
            put(k + "Names", num(r.names, 0)); put(k + "NamesDeclared", num(r.names_with_declaring_repo, 0))
            put(k + "Repos", num(r.repos_declaring, 0)); put(k + "MaxRepos", num(r.max_repos_one_name, 0))
            put(k + "Hits", num(r.search_hits, 0)); put(k + "Checked", num(r.files_checked, 0))


def robustness() -> None:
    import json as _j
    r = read("u10_outcome_robustness.csv")
    if r is None:
        return
    S = {"all": "All", "c1": "COne", "only2026": "Only"}
    for _, x in r.iterrows():
        k0 = S.get(x["set"], "")
        a = str(x.analysis)
        simple = {"primary": "Primary", "excl_probe_research": "NoProbe", "one_event_per_report_day": "PerDay",
                  "controls_no_drop": "NoDrop", "excl_prior_existence": "NoPrior",
                  "h6_events_before_model_release": "PreRelease", "excl_dotted_only": "NoDotted",
                  "npm_claimed_excl_spam_placeholder": "NpmClaimNoSpam", "npm_claimed_all": "NpmClaimAll",
                  "lineage_unrelated_models": "Lineage", "matched_sets": "Matched"}
        if a in simple:
            k = f"Rob{k0}{simple[a]}"
            put(k + "OR", num(x["or"], 1)); put(k + "Lo", num(x.or_lo, 1)); put(k + "Hi", num(x.or_hi, 1))
            put(k + "HK", num(x.h_events, 0)); put(k + "CK", num(x.c_events, 0))
            if isinstance(x.get("detail"), str) and x.detail.startswith("{"):
                for dk, dv in _j.loads(x.detail).items():
                    put(k + "".join(w.capitalize() for w in dk.split("_")), num(dv, 0))
        elif a.startswith("fragility"):
            put(f"Rob{k0}Fragility", num(float(x.detail), 0))
        elif a.startswith("kinds:"):
            d = _j.loads(x.detail)
            arm = "H" if a.endswith("halluc") else "C"
            for kk, v in d.items():
                put(f"Rob{k0}Kind{arm}" + "".join(w.capitalize() for w in kk.replace("+", "_").split("_")), num(v, 0))
    # E-value (VanderWeele & Ding 2017) for the pre-registered estimate; rare outcome, so OR approximates RR
    c1p = r[(r["set"] == "c1") & (r.analysis == "primary")]
    if len(c1p):
        ev = lambda x: x + (x * (x - 1)) ** 0.5 if x > 1 else 1.0
        put("EValue", num(ev(float(c1p["or"].iloc[0])), 1)); put("EValueLo", num(ev(float(c1p.or_lo.iloc[0])), 1))
    pr = r[(r["set"] == "all") & (r.analysis == "primary")]
    if len(pr):
        put("CtlPerName", num(pr.c_n.iloc[0] / pr.h_n.iloc[0], 1))
    src = r[(r["set"] == "all") & r.analysis.str.startswith("source:")]
    if len(src):
        put("RobSrcMinOR", num(src["or"].min(), 1)); put("RobSrcMaxOR", num(src["or"].max(), 1))
        put("RobSrcN", num(len(src), 0))
    srcc = r[(r["set"] == "c1") & r.analysis.str.startswith("source:")]
    if len(srcc):
        put("RobCOneSrcMinOR", num(srcc["or"].min(), 1)); put("RobCOneSrcMaxOR", num(srcc["or"].max(), 1))
    mo = r[r.analysis.str.startswith("model:")]
    if len(mo):
        put("RobModelMinOR", num(mo["or"].min(), 1)); put("RobModelMaxOR", num(mo["or"].max(), 1))
        put("RobModelN", num(len(mo), 0))
    s8 = read("u8_settings_rerun.csv")
    if s8 is not None:
        for _, x in s8.iterrows():
            k = "Set" + {"pooled": "", "npm": "Npm", "pypi": "Pypi"}[x.eco]
            put(k + "Prompts", num(x.prompts, 0)); put(k + "Before", pct(x.hall_share_before))
            put(k + "After", pct(x.hall_share_after)); put(k + "Diff", num(100 * x["diff"], 1))
            put(k + "DiffLo", num(100 * x.diff_lo, 1)); put(k + "DiffHi", num(100 * x.diff_hi, 1))
            put(k + "Identical", pct(x.identical_code_share, 0)); put(k + "Jaccard", num(x.mean_jaccard, 2))


def cross_ecosystem() -> None:
    x = read("u11_cross_ecosystem.csv")
    if x is None:
        return
    for _, r in x.iterrows():
        k = f"Cross{ECO[r.names_from]}To{ECO[r.registry]}"
        put(k + "N", num(r.cross_n, 0)); put(k + "K", num(r.cross_mal, 0)); put(k + "Rate", pct(r.cross_rate, 2))
        put(k + "HallRate", pct(r.halluc_rate, 2)); put(k + "CtlRate", pct(r.control_rate, 2))
        put(k + "VsCtl", num(r.rr_cross_vs_control, 1)); put(k + "HallVs", num(r.rr_halluc_vs_cross, 1))
        put(k + "PCtl", pval(r.p_cross_gt_control)); put(k + "PHall", pval(r.p_halluc_gt_cross))


def validation() -> None:
    v = read("u6_validation.csv")
    if v is None:
        return
    key = {"claimant class": "ValClass", "malicious vs not": "ValMal", "claimant class, inspectable pages": "ValClassSeen",
           "malicious vs not, inspectable pages": "ValMalSeen", "automatic unclear resolved by labeller": "ValUnclear",
           "automatic organic confirmed": "ValOrganic", "semantic real vs shuffled": "ValSem"}
    for _, r in v.iterrows():
        k = key.get(r.check)
        if not k:
            continue
        put(k + "N", num(r.n, 0))
        if pd.notna(r.get("agreement")):
            put(k + "Agree", pct(r.agreement, 0))
        if pd.notna(r.get("kappa")):
            put(k + "Kappa", num(r.kappa, 2))
        if k == "ValSem":
            put("ValSemReal", num(r.real_mean, 2)); put("ValSemShuf", num(r.shuffled_mean, 2))
            put("ValSemClear", pct(r.real_share_clear, 0)); put("ValSemP", pval(r.p_one_sided))
            put("ValSemAny", pct(r.real_share_any, 0)); put("ValSemShufAny", pct(r.shuffled_share_any, 0))
            put("ValSemNReal", num(r.n_real, 0)); put("ValSemNShuf", num(r.n_shuffled, 0))


def claims_summary() -> None:
    r1 = read("rq1_claim_rate.csv")
    if r1 is not None:
        for _, r in r1.iterrows():
            e = ECO[r.eco]
            put(f"Claim{e}Eligible", num(r.eligible, 0))
            put(f"Claim{e}Claimed", num(r.claimed, 0))
            put(f"Claim{e}Rate", pct(r.rate))
            put(f"Claim{e}Lo", pct(r.ci_lo))
            put(f"Claim{e}Hi", pct(r.ci_hi))
    s = TABLES / "summary.json"
    if s.exists():
        j = json.loads(s.read_text())
        for m, k in (("Lanyado huggingface-cli report", "Lanyado"), ("USENIX Security 2025", "Usenix")):
            v = (j.get("claims_after_milestone") or {}).get(m)
            put(f"ClaimsAfter{k}", num(v, 0) if v is not None else None)
        put("ClaimsDated", num(j.get("claims_with_known_date"), 0) if j.get("claims_with_known_date") else None)
    pm = read("per_model.csv")
    if pm is not None:
        hs = pm.groupby("model").hallucinated_share.mean()
        put("HallShareMin", pct(hs.min(), 0))
        put("HallShareMax", pct(hs.max(), 0))
        put("NModelsTwentyFour", num(pm.model.nunique(), 0))


def main() -> None:
    for f in (malware, conditional, dose, campaigns_timing_semantic, confounds, confirmatory,
              exposure_claimants_defense, detection_manifests, robustness, cross_ecosystem, validation, claims_summary):
        try:
            f()
        except Exception as ex:  # noqa: BLE001  (a table in an older format leaves its macros as ??)
            print(f"{f.__name__}: {type(ex).__name__}: {ex}")
    OUT.parent.mkdir(exist_ok=True)
    lines = ["% generated by src/15_numbers.py from tables/ -- do not edit by hand"]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    missing = [k for k, v in macros.items() if v == MISSING]
    print(f"{len(macros)} macros -> {OUT}; {len(missing)} not yet available")


if __name__ == "__main__":
    main()
