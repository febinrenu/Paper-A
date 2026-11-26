"""Robustness of the malware outcome (reviewer-requested), identical for hallucinated names and controls.

(a) Reporting source: OSV MAL records come from a few reporters. For each source, the odds ratio of "flagged by
    this source after the cutoff" for hallucinated names vs controls (Mantel-Haenszel over ecosystems). A detection
    bias towards hallucination-like names would show up as a source-specific excess, not a uniform one.
(b) Payload type: each record's details text is classified by keyword rules into credential/data theft,
    dropper/remote code, reconnaissance or dependency-confusion probe, research/proof of concept, or generic
    (no detail beyond "malicious code"). The odds ratio is recomputed excluding probes and research packages.
(c) Clustering: events reported on the same day are collapsed to one event per ecosystem and day, in both arms.
(d) Fragility: how many control non-events would have to be events for the pre-registered H1 test to lose
    significance.
(e) Prior existence (22_prior_existence.py): names with download history before the snapshot are removed from both
    arms.
(f) Control-drop rule: controls that a 2024-era model emitted removed (primary) vs all controls kept.
(g) Per discovery/confirmatory model: the malicious rate of each model's names vs the pooled controls.
(i) Lineage hold-out: C1 restricted to names from the two confirmatory models whose lineage is unrelated to any
    discovery model (Gemma 1.0, Qwen1.5), against the controls generated from those names.
(j) Matched sets: each hallucinated name compared only with its own controls (Mantel-Haenszel over matched sets,
    the conditional analysis for 1:K matching).
(h) H6 timing: names that only 2026 models emit, split by whether the first post-cutoff malware report predates the
    release of every model in our cohort that emits the name (hosted models and undated releases: 2026-01-01).
Outputs: tables/u10_outcome_robustness.csv (aggregates only).
"""
from __future__ import annotations

import importlib
import json
import re
import zipfile

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import StratifiedTable

from common import ALL_MODELS, DERIVED, REGISTRY, TABLES

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")

RULES = [
    ("research_poc", r"security research|proof[- ]of[- ]concept|\bpoc\b|bug bounty|for research|educational purpose|"
                     r"researcher|pentest|harmless"),
    ("probe", r"dependency confusion|confusion attack|system information|hostname|username|reconnaissance|"
              r"\bbeacon|fingerprint|out-of-band|\boast|interactsh|burpcollaborator|dns (?:request|query|lookup)"),
    ("theft", r"steal|exfiltrat|credential|password|token|cookie|wallet|crypto|discord|browser data|"
              r"keylog|clipboard|ssh key|environment variables|\.env|api key|seed phrase"),
    ("dropper", r"download(?:s|ed)? (?:and|then)? ?(?:execut|run)|second[- ]stage|"
                r"(?:download|fetch|drop)\w* (?:a |an |the )?(?:\w+ )?payload|reverse shell|backdoor|"
                r"remote (?:code|command)|execut(?:es|ing) (?:a |an )?(?:binary|script|command)|trojan|\brat\b"),
]


def records() -> pd.DataFrame:
    rows = []
    for eco, zname in (("pypi", "osv_PyPI_all.zip"), ("npm", "osv_npm_all.zip")):
        with zipfile.ZipFile(REGISTRY / zname) as z:
            for info in z.infolist():
                if not info.filename.startswith("MAL-"):
                    continue
                j = json.loads(z.read(info))
                if j.get("withdrawn"):
                    continue
                pub = pd.to_datetime(j.get("published"), utc=True, format="ISO8601")
                if pub <= st.CUTOFF:
                    continue
                srcs = sorted({o.get("source") or "unspecified"
                               for o in (j.get("database_specific") or {}).get("malicious-packages-origins") or []})
                text = ((j.get("summary") or "") + " " + (j.get("details") or "")).lower()
                for a in j.get("affected", []):
                    n = a.get("package", {}).get("name")
                    if n:
                        rows.append({"eco": eco, "name": n, "id": j["id"], "published": pub,
                                     "sources": srcs or ["unspecified"], "text": text})
    return pd.DataFrame(rows)


def kind(text: str) -> str:
    # drop the boilerplate every GHSA-sourced record carries ("...should be considered fully compromised...")
    t = re.sub(r"any computer that has this package installed.*?(?:software resulting from installing it\.)", " ", text,
               flags=re.S)
    for k, pat in RULES:
        if re.search(pat, t):
            return k
    return "generic"


def mh(h: pd.DataFrame, c: pd.DataFrame, hk: pd.Series, ck: pd.Series) -> dict:
    tabs, parts = [], {}
    for eco in ("npm", "pypi"):
        a, b = hk[h.eco == eco], ck[c.eco == eco]
        ak, an, bk, bn = int(a.sum()), len(a), int(b.sum()), len(b)
        parts[eco] = (ak, an, bk, bn)
        tabs.append(np.array([[ak, an - ak], [bk, bn - bk]]) + (0.5 if min(ak, bk) == 0 else 0))
    t = StratifiedTable(tabs)
    lo, hi = t.oddsratio_pooled_confint()
    return {"h_events": sum(p[0] for p in parts.values()), "h_n": sum(p[1] for p in parts.values()),
            "c_events": sum(p[2] for p in parts.values()), "c_n": sum(p[3] for p in parts.values()),
            "or": t.oddsratio_pooled, "or_lo": lo, "or_hi": hi, "p": t.test_null_odds().pvalue}


def main() -> None:
    cut = st.CUTOFF
    rec = records()
    rec["kind"] = rec.text.map(kind)
    out = []
    for which in ("all", "c1"):
        h, c = mech.hallucinated(cut, which), mech.controls(cut, which)
        key_h, key_c = list(zip(h.eco, h.name)), list(zip(c.eco, c.name))
        base = mh(h, c, h.mal_post, c.mal_post)
        out.append({"set": which, "analysis": "primary", **base})

        # (a) per source
        src = rec.explode("sources")
        for s, g in src.groupby("sources"):
            ks = set(zip(g.eco, g.name))
            hk = pd.Series([k in ks for k in key_h], index=h.index)
            ck = pd.Series([k in ks for k in key_c], index=c.index)
            if hk.sum() + ck.sum() >= 3:
                out.append({"set": which, "analysis": f"source:{s}", **mh(h, c, hk, ck)})
        # (b) payload kinds: counts and OR excluding probes / research
        kinds = rec.groupby(["eco", "name"]).kind.agg(lambda s: sorted(set(s)))
        for arm, d, keys in (("halluc", h, key_h), ("control", c, key_c)):
            ks = [k for k, m in zip(keys, d.mal_post) if m]
            cnt = pd.Series([("+".join(kinds.get(k, ["unknown"]))) for k in ks]).value_counts().to_dict()
            out.append({"set": which, "analysis": f"kinds:{arm}", "detail": json.dumps(cnt)})
        benign = {k for k, v in kinds.items() if set(v) <= {"probe", "research_poc"}}
        hk = pd.Series([m and k not in benign for k, m in zip(key_h, h.mal_post)], index=h.index)
        ck = pd.Series([m and k not in benign for k, m in zip(key_c, c.mal_post)], index=c.index)
        out.append({"set": which, "analysis": "excl_probe_research", **mh(h, c, hk, ck)})
        # (c) collapse same-day reports to one event per ecosystem-day, in both arms
        first = rec.groupby(["eco", "name"]).published.min().dt.date
        for arm, d, keys in (("h", h, key_h), ("c", c, key_c)):
            days = pd.Series([first.get(k) if m else None for k, m in zip(keys, d.mal_post)], index=d.index)
            seen, flag = set(), []
            for e, dd in zip(d.eco, days):
                if dd is None or (e, dd) in seen:
                    flag.append(False)
                else:
                    seen.add((e, dd)); flag.append(True)
            if arm == "h":
                hk = pd.Series(flag, index=d.index)
            else:
                ck = pd.Series(flag, index=d.index)
        out.append({"set": which, "analysis": "one_event_per_report_day", **mh(h, c, hk, ck)})
        # (e) prior existence removed from both arms
        pe = DERIVED / "prior_existence.parquet"
        if pe.exists():
            p = pd.read_parquet(pe)
            prior = set(zip(p.eco[p.prior_existence == True], p.name[p.prior_existence == True]))  # noqa: E712
            hm = pd.Series([k not in prior for k in key_h], index=h.index)
            cm = pd.Series([k not in prior for k in key_c], index=c.index)
            r = mh(h[hm], c[cm], h.mal_post[hm], c.mal_post[cm])
            out.append({"set": which, "analysis": "excl_prior_existence",
                        "detail": json.dumps({"h_removed": int((~hm).sum()), "c_removed": int((~cm).sum()),
                                              "h_events_removed": int((h.mal_post & ~hm).sum()),
                                              "c_events_removed": int((c.mal_post & ~cm).sum())}), **r})
        # (f) control-drop rule
        c0 = mech.controls(cut, which, drop="none")
        out.append({"set": which, "analysis": "controls_no_drop", **mh(h, c0, h.mal_post, c0.mal_post)})
        # pre-registered secondary reports: names that only appear as dotted module paths; npm claims without
        # spam and placeholder claims (claimants are classified for hallucinated names only, so this can only
        # lower the hallucinated claim rate)
        nd = ~h.only_dotted.fillna(False).astype(bool)
        out.append({"set": which, "analysis": "excl_dotted_only", **mh(h[nd], c, h.mal_post[nd], c.mal_post),
                    "detail": json.dumps({"removed": int((~nd).sum())})})
        cl = pd.read_parquet(DERIVED / "claimants.parquet")
        sp = set(zip(cl.eco[cl["class"].isin(["spam", "placeholder"])], cl.name[cl["class"].isin(["spam", "placeholder"])]))
        hn = h[h.eco == "npm"]
        cn = c[c.eco == "npm"]
        hk = hn.claimed & pd.Series([k not in sp for k in zip(hn.eco, hn.name)], index=hn.index)
        a, b = [[int(hk.sum()), len(hn) - int(hk.sum())], [int(cn.claimed.sum()), len(cn) - int(cn.claimed.sum())]]
        orr, pv = stats.fisher_exact([a, b])
        out.append({"set": which, "analysis": "npm_claimed_excl_spam_placeholder", "h_events": a[0], "h_n": len(hn),
                    "c_events": b[0], "c_n": len(cn), "or": orr, "p": pv,
                    "detail": json.dumps({"removed": int((hn.claimed & ~hk).sum())})})
        orr, pv = stats.fisher_exact([[int(hn.claimed.sum()), len(hn) - int(hn.claimed.sum())], b])
        out.append({"set": which, "analysis": "npm_claimed_all", "h_events": int(hn.claimed.sum()), "h_n": len(hn),
                    "c_events": b[0], "c_n": len(cn), "or": orr, "p": pv})
        # (d) fragility of the pooled one-sided Fisher test (H1 decision rule)
        ak, an, bk, bn = base["h_events"], base["h_n"], base["c_events"], base["c_n"]
        f = 0
        while stats.fisher_exact([[ak, an - ak], [bk + f, bn - bk - f]], alternative="greater")[1] < 0.05:
            f += 1
        out.append({"set": which, "analysis": "fragility_control_events_to_lose_p05", "detail": str(f)})
    # (g) per model, pooled over ecosystems, against all controls
    h, c = mech.hallucinated(cut, "all"), mech.controls(cut, "all")
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")
    m24 = mentions[mentions.era == 2024][["model", "eco", "name"]].drop_duplicates()
    for model, g in m24.groupby("model"):
        ks = set(zip(g.eco, g.name))
        hm = pd.Series([k in ks for k in zip(h.eco, h.name)], index=h.index)
        out.append({"set": "all", "analysis": f"model:{model}", **mh(h[hm], c, h.mal_post[hm], c.mal_post)})
    # (i) lineage hold-out
    h1, c1 = mech.hallucinated(cut, "c1"), mech.controls(cut, "c1")
    unrelated = {"gemma:7b-instruct-q4_0", "qwen:7b-chat"}
    em = mentions[mentions.model.isin(unrelated)][["eco", "name"]].drop_duplicates()
    ks = set(zip(em.eco, em.name))
    hm = pd.Series([k in ks for k in zip(h1.eco, h1.name)], index=h1.index)
    ctl1 = pd.read_parquet(DERIVED / "controls_c1.parquet")[["eco", "name", "matched_to"]]
    keep = set(zip(h1.eco[hm], h1.name[hm]))
    cnames = {(e, n) for e, n, m in zip(ctl1.eco, ctl1.name, ctl1.matched_to) if (e, m) in keep}
    cm = pd.Series([k in cnames for k in zip(c1.eco, c1.name)], index=c1.index)
    out.append({"set": "c1", "analysis": "lineage_unrelated_models", **mh(h1[hm], c1[cm], h1.mal_post[hm], c1.mal_post[cm])})
    # (j) matched sets: MH over (hallucinated name + its own controls)
    for which, files in (("all", ["controls.parquet", "controls_c1.parquet"]), ("c1", ["controls_c1.parquet"])):
        hh, cc = mech.hallucinated(cut, which), mech.controls(cut, which)
        link = pd.concat([pd.read_parquet(DERIVED / f)[["eco", "name", "matched_to"]] for f in files]).drop_duplicates(["eco", "name"])
        if "matched_to" not in cc:
            cc = cc.merge(link, on=["eco", "name"], how="left")
        hset = dict(zip(zip(hh.eco, hh.name), hh.mal_post))
        tabs, sets, ev = [], 0, 0
        for (e, m), g in cc.groupby(["eco", "matched_to"]):
            if (e, m) not in hset:
                continue
            a = int(hset[(e, m)]); b = int(g.mal_post.sum())
            sets += 1
            if a + b == 0:
                continue                      # concordant sets carry no information in the conditional analysis
            ev += a + b
            tabs.append(np.array([[a, 1 - a], [b, len(g) - b]]))
        t = StratifiedTable(tabs)
        lo, hi = t.oddsratio_pooled_confint()
        out.append({"set": which, "analysis": "matched_sets", "or": t.oddsratio_pooled, "or_lo": lo, "or_hi": hi,
                    "p": t.test_null_odds().pvalue, "h_events": sum(int(x[0, 0]) for x in tabs),
                    "c_events": sum(int(x[1, 0]) for x in tabs),
                    "detail": json.dumps({"matched_sets": sets, "informative_sets": len(tabs)})})
    # (h) H6 split by model release
    cand, _ = st.load_candidates(cut)
    only26 = cand[(cand.n_models_2024 == 0) & (cand.status != "pre_cutoff")
                  & ((cand.n_models_2026 > 0) | (cand.n_models_2026h > 0))]
    first = rec.groupby(["eco", "name"]).published.min()
    m26 = mentions[mentions.era == 2026][["model", "eco", "name"]].drop_duplicates()
    rel = {k: pd.Timestamp(v["release"] if v.get("release") and len(str(v["release"])) > 4 else "2026-01-01", tz="UTC")
           for k, v in ALL_MODELS.items()}
    earliest = m26.assign(rel=m26.model.map(rel)).groupby(["eco", "name"]).rel.min()
    ev = [k for k in zip(only26.eco, only26.name) if k in first.index]
    pre = [k for k in ev if first[k] < earliest.get(k, pd.Timestamp("2026-01-01", tz="UTC"))]
    hp = pd.Series([k in set(pre) for k in zip(only26.eco, only26.name)], index=only26.index)
    r = mh(only26, c, hp, c.mal_post)
    out.append({"set": "only2026", "analysis": "h6_events_before_model_release", **r,
                "detail": json.dumps({"events": len(ev), "before_release": len(pre)})})
    o = pd.DataFrame(out)
    o.to_csv(TABLES / "u10_outcome_robustness.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 120):
        print(o.to_string(index=False))


if __name__ == "__main__":
    main()
