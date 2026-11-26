"""Further robustness checks requested in review; every rule is applied identically to both arms.

(a) Dependency-confusion fingerprint: malicious events whose affected versions include a major version >= 90 (the
    usual signature of dependency-confusion packages, which target internal names seen in public code rather than
    names recommended by models) are dropped from both arms. Versions come from the OSV records.
(b) Campaign independence of C1: C1 events (and C1 control events) that share an OSV report day with a discovery
    malicious name, or lie within edit distance 2 of one, are dropped.
(c) Registrable names: npm refuses a new name that equals an existing name once punctuation (. _ -) is removed. Names
    colliding with a 2024-snapshot name in that sense are removed from both arms (npm).
(d) Detection period: events reported before vs from April 2025 (after 'slopsquatting' was coined), separately.
(e) Prompt step: hallucinated names that came from the code or the 'required packages' answer vs names that only
    came from the 'useful packages' answer, each against all controls.
(f) Positivity: hallucinated names with no surviving control.
(g) H5: current state of the malicious names that 2026 models still emit (free to re-register vs held).
(h) Cross-registry baseline with confidence intervals, pooled, and restricted to ecosystem-neutral names.
Outputs: tables/u12_reviewer_checks.csv (aggregates only).
"""
from __future__ import annotations

import importlib
import json
import re

import httpx
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

from common import DERIVED, REGISTRY, TABLES, USER_AGENT

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")
rob = importlib.import_module("23_outcome_robustness")
CACHE = DERIVED / "osv_record_cache.jsonl"


def mh(h, c, hk, ck) -> dict:
    return rob.mh(h, c, hk, ck)


def osv_versions(ids: list[str], cache: dict, client) -> list[str]:
    out = set()
    for i in ids:
        if i not in cache:
            r = client.get(f"https://api.osv.dev/v1/vulns/{i}")
            cache[i] = r.json() if r.status_code == 200 else {}
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps({"id": i, "rec": cache[i]}) + "\n")
        j = cache[i]
        for a in j.get("affected", []):
            out |= set(a.get("versions") or [])
        for o in (j.get("database_specific") or {}).get("malicious-packages-origins", []) or []:
            out |= set(o.get("versions") or [])
    return sorted(out)


def major(v: str) -> int:
    m = re.match(r"\D*(\d+)", v)
    return int(m.group(1)) if m else -1


def edit_le2(a: str, b: str) -> bool:
    if abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        if min(cur) > 2:
            return False
        prev = cur
    return prev[-1] <= 2


def katz(k1, n1, k0, n0):
    if not (k1 and k0):
        return (np.nan, np.nan, np.nan)
    rr = (k1 / n1) / (k0 / n0)
    se = np.sqrt(1 / k1 - 1 / n1 + 1 / k0 - 1 / n0)
    return rr, rr * np.exp(-1.96 * se), rr * np.exp(1.96 * se)


def main() -> None:
    cut = st.CUTOFF
    out = []
    rec = mech.osv_records()
    rec = rec[rec.published > cut]
    ids = rec.groupby(["eco", "name"]).id.apply(list)
    first = rec.groupby(["eco", "name"]).published.min()
    cache = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            cache[d["id"]] = d["rec"]
    client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60)

    for which in ("all", "c1"):
        h, c = mech.hallucinated(cut, which), mech.controls(cut, which)
        base = mh(h, c, h.mal_post, c.mal_post)
        out.append({"set": which, "analysis": "primary", **base})
        # (a) dependency-confusion fingerprint
        fp = {}
        for d in (h, c):
            for e, n in zip(d.eco[d.mal_post], d.name[d.mal_post]):
                vs = osv_versions(ids.get((e, n), []), cache, client)
                fp[(e, n)] = any(major(v) >= 90 for v in vs)
        hk = pd.Series([m and not fp.get(k, False) for k, m in zip(zip(h.eco, h.name), h.mal_post)], index=h.index)
        ck = pd.Series([m and not fp.get(k, False) for k, m in zip(zip(c.eco, c.name), c.mal_post)], index=c.index)
        out.append({"set": which, "analysis": "excl_depconf_fingerprint", **mh(h, c, hk, ck),
                    "detail": json.dumps({"h_flagged": int((h.mal_post & ~hk).sum()), "c_flagged": int((c.mal_post & ~ck).sum())})})
        # (c) registrable on npm
        snap_npm = set((REGISTRY / "npm_2024.txt").read_text(encoding="utf-8").split("\n"))
        strip = lambda n: re.sub(r"[._-]", "", n)
        stripped = {}
        for n in snap_npm:
            stripped.setdefault(strip(n), n)
        def blocked(e, n):
            if e != "npm":
                return False
            s = stripped.get(strip(n))
            return s is not None and s != n
        hb = pd.Series([blocked(e, n) for e, n in zip(h.eco, h.name)], index=h.index)
        cb = pd.Series([blocked(e, n) for e, n in zip(c.eco, c.name)], index=c.index)
        out.append({"set": which, "analysis": "registrable_only", **mh(h[~hb], c[~cb], h.mal_post[~hb], c.mal_post[~cb]),
                    "detail": json.dumps({"h_blocked": int(hb.sum()), "c_blocked": int(cb.sum()),
                                          "h_blocked_events": int((hb & h.mal_post).sum())})})
        # (d) detection period
        split = pd.Timestamp("2025-04-12", tz="UTC")
        for lab, f in (("reported_before_2025_04", lambda t: t < split), ("reported_from_2025_04", lambda t: t >= split)):
            hk = pd.Series([m and f(first[k]) for k, m in zip(zip(h.eco, h.name), h.mal_post)], index=h.index)
            ck = pd.Series([m and f(first[k]) for k, m in zip(zip(c.eco, c.name), c.mal_post)], index=c.index)
            out.append({"set": which, "analysis": lab, **mh(h, c, hk, ck)})
        # (e) prompt step
        names = pd.read_parquet(DERIVED / "names.parquet")[["eco", "name", "sources"]]
        hs = h if "sources" in h else h.merge(names, on=["eco", "name"], how="left")
        req = hs.sources.fillna("").str.contains("code|t1")
        for lab, m in (("from_code_or_required", req), ("useful_only", ~req)):
            g = hs[m.values]
            out.append({"set": which, "analysis": lab, **mh(g, c, g.mal_post, c.mal_post)})
        # (f) positivity
        files = {"all": ["controls.parquet", "controls_c1.parquet"], "c1": ["controls_c1.parquet"]}[which]
        link = pd.concat([pd.read_parquet(DERIVED / f)[["eco", "name", "matched_to"]] for f in files])
        cm = c.merge(link.drop_duplicates(["eco", "name"]), on=["eco", "name"], how="left") if "matched_to" not in c else c
        has = set(zip(cm.eco, cm.matched_to))
        no = pd.Series([k not in has for k in zip(h.eco, h.name)], index=h.index)
        out.append({"set": which, "analysis": "no_surviving_control",
                    "detail": json.dumps({"names": int(no.sum()), "share": float(no.mean()),
                                          "events": int((no & h.mal_post).sum())}),
                    **mh(h[~no], c, h.mal_post[~no], c.mal_post)})

    # (b) campaign independence of C1 from discovery events
    hd, h1, c1 = mech.hallucinated(cut, "discovery"), mech.hallucinated(cut, "c1"), mech.controls(cut, "c1")
    dmal = list(zip(hd.eco[hd.mal_post], hd.name[hd.mal_post]))
    dday = {(e, first[(e, n)].date()) for e, n in dmal}
    def linked(e, n):
        return (e, first[(e, n)].date()) in dday or any(e == e2 and edit_le2(n, n2) for e2, n2 in dmal)
    hk = pd.Series([m and not linked(e, n) for e, n, m in zip(h1.eco, h1.name, h1.mal_post)], index=h1.index)
    ck = pd.Series([m and not linked(e, n) for e, n, m in zip(c1.eco, c1.name, c1.mal_post)], index=c1.index)
    out.append({"set": "c1", "analysis": "campaign_independent", **mh(h1, c1, hk, ck),
                "detail": json.dumps({"h_dropped": int((h1.mal_post & ~hk).sum()), "c_dropped": int((c1.mal_post & ~ck).sum())})})

    # (g) H5 names: current state
    cand, _ = st.load_candidates(cut)
    post = mech.osv_post(cut)
    live = cand[(cand.n_models_2024 > 0) & ((cand.n_models_2026 > 0) | (cand.n_models_2026h > 0))]
    live = live[[(e, n) in post for e, n in zip(live.eco, live.name)]]
    state = np.where(live.security_holding.fillna(False).astype(bool), "npm_placeholder",
                     np.where(live.status == "claimed_live", "registered_now", "deleted_free"))
    out.append({"set": "h5", "analysis": "h5_state", "detail": json.dumps(pd.Series(state).value_counts().to_dict())})

    # (h) cross-registry with CIs and ecosystem-neutral restriction
    x = importlib.import_module("24_cross_ecosystem")
    neutral_bad = re.compile(r"(^|[-_.])(py|python|pip|django|flask|fastapi|pandas|numpy|scipy|torch|tensorflow|"
                             r"pytest|sklearn|jupyter|node|npm|js|javascript|ts|typescript|react|vue|angular|svelte|"
                             r"next|nuxt|express|webpack|babel|eslint|jquery|vite|deno)([-_.]|$)")
    h, c = mech.hallucinated(cut, "all"), mech.controls(cut, "all")
    names = pd.read_parquet(DERIVED / "names.parquet")
    emitted = {e: set(names.name[names.eco == e]) for e in ("npm", "pypi")}
    snap = {"pypi": set((REGISTRY / "pypi_2024.txt").read_text(encoding="utf-8").split("\n")),
            "npm": set((REGISTRY / "npm_2024.txt").read_text(encoding="utf-8").split("\n"))}
    tabs_all, tabs_neu = [], []
    for src, dst in (("pypi", "npm"), ("npm", "pypi")):
        hs = h[h.eco == src]
        cand_n = hs.name if src == "pypi" else hs.name[~hs.name.str.startswith("@")]
        cross = {x.pep503(n) for n in cand_n}
        cross = {n for n in cross if n not in snap[dst] and n not in emitted[dst]}
        cd = c[c.eco == dst]
        ck_, cn_ = int(cd.mal_post.sum()), len(cd)
        for lab, S, tabs in (("all", cross, tabs_all), ("neutral", {n for n in cross if not neutral_bad.search(n)}, tabs_neu)):
            k = sum((dst, n) in post for n in S)
            rr, lo, hi = katz(k, len(S), ck_, cn_)
            out.append({"set": f"cross_{src}_to_{dst}", "analysis": f"cross_{lab}", "h_events": k, "h_n": len(S),
                        "c_events": ck_, "c_n": cn_, "or": rr, "or_lo": lo, "or_hi": hi})
            tabs.append(np.array([[k, len(S) - k], [ck_, cn_ - ck_]]))
    for lab, tabs in (("cross_pooled_all", tabs_all), ("cross_pooled_neutral", tabs_neu)):
        t = StratifiedTable(tabs)
        lo, hi = t.oddsratio_pooled_confint()
        out.append({"set": "cross", "analysis": lab, "or": t.oddsratio_pooled, "or_lo": lo, "or_hi": hi,
                    "p": t.test_null_odds().pvalue, "h_events": int(sum(x[0, 0] for x in tabs)),
                    "h_n": int(sum(x[0].sum() for x in tabs))})
    o = pd.DataFrame(out)
    o.to_csv(TABLES / "u12_reviewer_checks.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_colwidth", 100):
        print(o.to_string(index=False))


if __name__ == "__main__":
    main()
