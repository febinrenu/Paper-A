"""Exposure windows of malicious hallucinated packages: how long was the malicious code installable?

For every hallucinated name (2024-era cohort) with a post-cutoff, non-withdrawn OSV MAL record:
  * affected versions  from the OSV record (affected[].versions and the per-source version lists)
  * npm   publish time of the first affected version, from the packument's `time` map (npm keeps the
          timestamps of unpublished versions); removal = the security-placeholder version time, or the
          unpublish time of a tombstone, or, when the package was rolled back, the next packument
          modification after the last affected version
  * PyPI  not timed: deleted projects leave no metadata, and the public download logs continue after deletion
          and mix earlier incarnations of a name
  * report = first post-cutoff publication of the OSV record
Outputs: data/derived/detection.parquet (name-level, not published), tables/u2b_detection.csv (aggregates).
"""
from __future__ import annotations

import importlib
import json
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from common import DERIVED, TABLES, USER_AGENT

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")
CACHE = DERIVED / "detection_cache.jsonl"
CH = "https://sql-clickhouse.clickhouse.com/?user=demo"
c = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True)


def cached() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out[d["key"]] = d["val"]
    return out


def remember(key: str, val) -> None:
    with open(CACHE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "val": val}) + "\n")


def osv_versions(ids: list[str], cache: dict) -> list[str]:
    vers: set[str] = set()
    for i in ids:
        k = f"osv:{i}"
        if k not in cache:
            r = c.get(f"https://api.osv.dev/v1/vulns/{i}")
            cache[k] = r.json() if r.status_code == 200 else None
            remember(k, cache[k])
        j = cache[k] or {}
        for a in j.get("affected", []):
            vers |= set(a.get("versions") or [])
        for o in (j.get("database_specific") or {}).get("malicious-packages-origins", []) or []:
            vers |= set(o.get("versions") or [])
    return sorted(vers)


def npm_times(name: str, cache: dict) -> dict:
    k = f"npm:{name}"
    if k not in cache:
        r = c.get(f"https://registry.npmjs.org/{quote(name, safe='@/')}")
        cache[k] = r.json().get("time", {}) if r.status_code == 200 else {}
        remember(k, cache[k])
    return cache[k]


def pypi_window(names: list[str], cache: dict) -> dict:
    todo = [n for n in names if f"pypi:{n}" not in cache]
    if todo:
        lst = ",".join("'" + n.replace("'", "") + "'" for n in todo)
        q = (f"SELECT project, min(date), max(date) FROM pypi.pypi_downloads_per_day WHERE project IN ({lst}) "
             f"AND date >= '2024-01-01' GROUP BY project FORMAT TSV")
        r = c.post(CH, content=q.encode())
        got = {}
        if r.status_code == 200:
            for line in r.text.splitlines():
                p, a, b = line.split("\t")
                got[p] = [a, b]
            for n in todo:
                cache[f"pypi:{n}"] = got.get(n)
                remember(f"pypi:{n}", got.get(n))
    return {n: cache.get(f"pypi:{n}") for n in names}


def ts(x):
    return pd.to_datetime(x, utc=True, errors="coerce", format="ISO8601")


def main() -> None:
    cache = cached()
    cand, _ = st.load_candidates(st.CUTOFF)
    h = cand[(cand.n_models_2024 > 0) & (cand.status != "pre_cutoff")]
    recs = mech.osv_records()
    recs = recs[recs.published > st.CUTOFF]
    h = h.merge(recs.groupby(["eco", "name"]).agg(ids=("id", list), report=("published", "min")).reset_index(),
                on=["eco", "name"])
    rows = []
    for r in h.itertuples():
        vers = osv_versions(r.ids, cache)
        row = {"eco": r.eco, "name": r.name, "report": r.report, "n_affected_versions": len(vers),
               "start": pd.NaT, "removed": pd.NaT, "method": None}
        if r.eco == "npm":
            t = npm_times(r.name, cache)
            vt = sorted(ts(t[v]) for v in vers if v in t)
            row["method_start"] = "affected_versions" if vt else "post_cutoff_versions"
            if not vt:     # whole-package records without a version list: versions published after the cutoff
                vt = sorted(x for x in (ts(v) for k, v in t.items() if k not in ("created", "modified", "unpublished")
                                        and not k.endswith("-security")) if st.CUTOFF < x <= r.report)
            if vt:
                row["start"] = vt[0]
                sec = [ts(v) for k, v in t.items() if k.endswith("-security")]
                unp = (t.get("unpublished") or {}).get("time") if isinstance(t.get("unpublished"), dict) else None
                after = sorted(ts(v) for k, v in t.items()
                               if k not in ("created", "unpublished") and ts(v) > vt[-1] and k not in vers)
                if sec:
                    row["removed"], row["method"] = min(sec), "security_placeholder"
                elif unp:
                    row["removed"], row["method"] = ts(unp), "unpublished"
                elif after:
                    row["removed"], row["method"] = after[0], "rolled_back"
                else:
                    row["method"] = "no_removal_time"
        else:
            # PyPI deletes malicious projects entirely, and the public download logs continue after deletion and
            # mix earlier incarnations of a name, so no defensible start or removal time exists: not timed.
            row["method"] = "not_measurable"
        rows.append(row)
    d = pd.DataFrame(rows)
    for col in ("report", "start", "removed"):
        d[col] = ts(d[col])
    d["report_days"] = ((d.report - d.start).dt.total_seconds() / 86400).clip(lower=0)
    d["window_days"] = ((d.removed - d.start).dt.total_seconds() / 86400).clip(lower=0)
    d.to_parquet(DERIVED / "detection.parquet", index=False)

    out = []
    for eco, g in d.groupby("eco"):
        aff = g[g.get("method_start", pd.Series(index=g.index, dtype=object)) == "affected_versions"]
        for m, x in (("report_days", g.report_days), ("window_days", g.window_days),
                     ("window_days_affected_only", aff.window_days)):
            x = x.dropna()
            out.append({"eco": eco, "measure": m, "n": len(x), "of": len(g), "median": x.median(),
                        "p25": x.quantile(0.25), "p75": x.quantile(0.75), "max": x.max(),
                        "share_le_1d": (x <= 1).mean(), "share_gt_7d": (x > 7).mean(), "share_gt_30d": (x > 30).mean()})
        rr = g.dropna(subset=["removed", "report"])
        if len(rr):
            gap = (rr.report - rr.removed).abs().dt.total_seconds()
            out.append({"eco": eco, "measure": "report_vs_removal", "n": len(rr), "share_le_10min": (gap <= 600).mean(),
                        "max_names_same_report_hour": int(rr.report.dt.floor("h").value_counts().max())})
        out.append({"eco": eco, "measure": "methods", **g.method.value_counts(dropna=False).to_dict(),
                    **(g.method_start.value_counts().to_dict() if "method_start" in g else {})})
    o = pd.DataFrame(out)
    o.to_csv(TABLES / "u2b_detection.csv", index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()
