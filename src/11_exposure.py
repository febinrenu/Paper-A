"""U2 harm: did claimed hallucinated names reach anyone? Downloads, dependents and takedown latency.

For every claimed name (hallucinated by a 2024-era model, registered after the cutoff) and, as the
baseline, every claimed control name:
  * downloads  npm: api.npmjs.org/downloads/range (<= 540-day chunks); PyPI: ClickHouse public
               dataset pypi.pypi_downloads_per_day (full history). Raw download counts include mirrors
               and bots, so comparisons are made against claimed controls, which are subject to the
               same noise.
  * dependents deps.dev v3alpha :dependents for the default version (direct / indirect counts)
  * takedown   created -> first OSV MAL publication, and npm unpublish time where present
Outputs: data/derived/exposure.parquet (name-level, not published), tables/u2_exposure_summary.csv.
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

from common import DERIVED, TABLES, USER_AGENT

st = importlib.import_module("07_stats")
END = dt.date(2026, 11, 23)
CACHE = DERIVED / "exposure_cache.jsonl"
CH = "https://sql-clickhouse.clickhouse.com/?user=demo"
c = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=90, follow_redirects=True)


def get(url, **kw):
    for attempt in range(8):
        try:
            r = c.get(url, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(20 * (attempt + 1))
                continue
            return r
        except (httpx.ConnectError, httpx.ConnectTimeout):
            time.sleep(60)
        except httpx.HTTPError:
            time.sleep(10 * (attempt + 1))
    return None


def npm_downloads(name: str, start: dt.date) -> dict:
    total, days_pos, first_day = 0, 0, None
    s = max(start, dt.date(2015, 1, 10))
    while s <= END:
        e = min(s + dt.timedelta(days=539), END)
        r = get(f"https://api.npmjs.org/downloads/range/{s}:{e}/{quote(name, safe='@/')}")
        if r is not None and r.status_code == 200:
            for d in r.json().get("downloads", []):
                if d["downloads"]:
                    total += d["downloads"]
                    days_pos += 1
                    first_day = first_day or d["day"]
        s = e + dt.timedelta(days=1)
    return {"downloads": total, "days_with_downloads": days_pos, "first_download_day": first_day}


def pypi_downloads(names: list[str]) -> dict:
    out = {}
    for i in range(0, len(names), 150):
        chunk = names[i:i + 150]
        lst = ",".join("'" + n.replace("'", "") + "'" for n in chunk)
        q = (f"SELECT project, sum(count), countIf(count > 0), min(date) FROM pypi.pypi_downloads_per_day "
             f"WHERE project IN ({lst}) AND date >= '2024-01-01' GROUP BY project FORMAT TSV")
        for attempt in range(6):
            try:
                r = c.post(CH, content=q.encode())
                if r.status_code == 200:
                    for line in r.text.splitlines():
                        p, tot, days, first = line.split("\t")
                        out[p] = {"downloads": int(tot), "days_with_downloads": int(days), "first_download_day": first}
                    break
            except httpx.HTTPError:
                pass
            time.sleep(30 * (attempt + 1))
    return {n: out.get(n, {"downloads": 0, "days_with_downloads": 0, "first_download_day": None}) for n in names}


def dependents(eco: str, name: str) -> dict:
    system = "npm" if eco == "npm" else "pypi"
    r = get(f"https://api.deps.dev/v3/systems/{system}/packages/{quote(name, safe='')}")
    if r is None or r.status_code != 200:
        return {"dependents": None, "direct_dependents": None}
    vers = r.json().get("versions", [])
    v = next((x for x in vers if x.get("isDefault")), vers[-1] if vers else None)
    if not v:
        return {"dependents": None, "direct_dependents": None}
    ver = v["versionKey"]["version"]
    d = get(f"https://api.deps.dev/v3alpha/systems/{system}/packages/{quote(name, safe='')}/versions/{quote(ver, safe='')}:dependents")
    if d is None or d.status_code != 200:
        return {"dependents": None, "direct_dependents": None}
    j = d.json()
    return {"dependents": j.get("dependentCount"), "direct_dependents": j.get("directDependentCount")}


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out[(d["eco"], d["name"])] = d
    return out


def main() -> None:
    cand, _ = st.load_candidates(st.CUTOFF)
    h = cand[(cand.n_models_2024 > 0) & cand.claimed].assign(group="hallucinated")
    ctl = pd.read_parquet(DERIVED / "controls.parquet")
    ctl["created_ts"] = pd.to_datetime(ctl.created, utc=True, errors="coerce")
    ctl = ctl[(ctl.exists | ctl.tombstone) & ~(ctl.created_ts.notna() & (ctl.created_ts <= st.CUTOFF))].assign(group="control")
    rows = pd.concat([h[["eco", "name", "group", "created", "malicious", "mal_first_published", "status"]],
                      ctl[["eco", "name", "group", "created"]]], ignore_index=True)
    rows["malicious"] = rows.malicious.fillna(False).astype(bool)
    cache = load_cache()
    todo = rows[[(e, n) not in cache for e, n in zip(rows.eco, rows.name)]]
    print(f"{len(rows)} claimed names ({(rows.group == 'hallucinated').sum()} hallucinated), {len(todo)} to fetch", flush=True)

    py = sorted(set(todo.name[todo.eco == "pypi"]))
    pyd = pypi_downloads(py) if py else {}

    def one(r):
        created = pd.to_datetime(r.created, utc=True, errors="coerce")
        start = created.date() if not pd.isna(created) else dt.date(2024, 4, 1)
        d = npm_downloads(r.name, start) if r.eco == "npm" else pyd[r.name]
        return {"eco": r.eco, "name": r.name, **d, **dependents(r.eco, r.name)}

    with ThreadPoolExecutor(6) as ex:
        for i, res in enumerate(ex.map(one, todo.itertuples()), 1):
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(res) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    cache = load_cache()
    ex_df = pd.DataFrame([{**r._asdict(), **cache.get((r.eco, r.name), {})} for r in rows.itertuples(index=False)])
    created = pd.to_datetime(ex_df.created, utc=True, errors="coerce")
    malpub = pd.to_datetime(ex_df.mal_first_published, utc=True, errors="coerce")
    ex_df["takedown_days"] = (malpub - created).dt.total_seconds() / 86400
    ex_df.to_parquet(DERIVED / "exposure.parquet", index=False)

    def summ(g):
        d = g.downloads.fillna(0)
        return pd.Series({"n": len(g), "downloads_median": d.median(), "downloads_p90": d.quantile(0.9),
                          "downloads_total": d.sum(), "share_ge_1000": (d >= 1000).mean(),
                          "share_with_dependents": (g.dependents.fillna(0) > 0).mean(),
                          "dependents_total": g.dependents.fillna(0).sum(),
                          "takedown_days_median": g.takedown_days.median()})
    ex_df["subgroup"] = np.where(ex_df.group == "control", "control_claimed",
                                 np.where(ex_df.malicious, "hallucinated_malicious", "hallucinated_other"))
    out = ex_df.groupby(["eco", "subgroup"]).apply(summ).reset_index()
    out.to_csv(TABLES / "u2_exposure_summary.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
