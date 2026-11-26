"""Did a name exist (and get deleted) before the 2024 snapshot? Applied identically to hallucinated names and controls.

A name absent from the early-2024 snapshot may still have been a real package that was deleted before it, in which
case a model may have memorised it rather than invented it, and a later malicious registration may be a "revival"
of a deleted name rather than a hallucination being targeted. Registry metadata of deleted projects is gone, but
download logs are not:
  * PyPI  public ClickHouse copy of the PyPI download logs (pypi.pypi_downloads_per_day, from 2016)
  * npm   (not used, see main(); kept for the record) api.npmjs.org bulk download ranges (unscoped names, 128 per request; scoped names one by one); the API
          serves counts only from 2021, so npm deletions before 2021 are not detected (npm tombstones are, via 07_stats)
A name with any download before the snapshot (PyPI 2024-03-12, npm 2024-01-15) is flagged "prior_existence".
Outputs: data/derived/prior_existence.parquet (name-level, not published).
"""
from __future__ import annotations

import datetime as dt
import importlib
import json
import time
from urllib.parse import quote

import httpx
import pandas as pd

from common import DERIVED, USER_AGENT

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")
CH = "https://sql-clickhouse.clickhouse.com/?user=demo"
CACHE = DERIVED / "prior_existence_cache.jsonl"
SNAP = {"pypi": "2024-03-12", "npm": dt.date(2024, 1, 15)}
c = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120, follow_redirects=True)


def post(q: str) -> str | None:
    for attempt in range(8):
        try:
            r = c.post(CH, content=q.encode())
            if r.status_code == 200:
                return r.text
        except httpx.HTTPError:
            pass
        time.sleep(15 * (attempt + 1))
    return None


def get(url: str):
    for attempt in range(8):
        try:
            r = c.get(url)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return {}
            if r.status_code == 429:
                time.sleep(30 * (attempt + 1))
                continue
        except (httpx.HTTPError, ValueError):
            pass
        time.sleep(10 * (attempt + 1))
    return None


def load() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out[(d["eco"], d["name"])] = d["prior"]
    return out


def save(rows: list[dict]) -> None:
    with open(CACHE, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def pypi(names: list[str]) -> None:
    for i in range(0, len(names), 300):
        chunk = names[i:i + 300]
        lst = ",".join("'" + n.replace("'", "") + "'" for n in chunk)
        txt = post(f"SELECT DISTINCT project FROM pypi.pypi_downloads_per_day WHERE project IN ({lst}) "
                   f"AND date < '{SNAP['pypi']}' FORMAT TSV")
        if txt is None:
            print("clickhouse failed; will retry next run", flush=True)
            continue
        seen = set(txt.split())
        save([{"eco": "pypi", "name": n, "prior": n in seen} for n in chunk])
        if (i // 300) % 20 == 0:
            print(f"  pypi {i + len(chunk)}/{len(names)}", flush=True)


def npm_ranges():
    s = dt.date(2021, 1, 1)          # api.npmjs.org serves no counts for earlier periods (verified on a top package)
    while s < SNAP["npm"]:
        e = min(s + dt.timedelta(days=364), SNAP["npm"] - dt.timedelta(days=1))
        yield s, e
        s = e + dt.timedelta(days=1)


def npm(names: list[str]) -> None:
    unscoped = [n for n in names if not n.startswith("@")]
    scoped = [n for n in names if n.startswith("@")]
    batches = [unscoped[i:i + 128] for i in range(0, len(unscoped), 128)]
    for bi, b in enumerate(batches):
        prior = {n: False for n in b}
        ok = True
        for s, e in npm_ranges():
            todo = [n for n in b if not prior[n]]
            if not todo:
                break
            j = get(f"https://api.npmjs.org/downloads/point/{s}:{e}/{','.join(quote(n, safe='') for n in todo)}")
            if j is None:
                ok = False
                break
            if len(todo) == 1:
                j = {todo[0]: j}
            for n in todo:
                v = j.get(n) if isinstance(j, dict) else None
                if isinstance(v, dict) and (v.get("downloads") or 0) > 0:
                    prior[n] = True
            time.sleep(0.3)
        if ok:
            save([{"eco": "npm", "name": n, "prior": p} for n, p in prior.items()])
        if bi % 20 == 0:
            print(f"  npm batch {bi + 1}/{len(batches)}", flush=True)
    # scoped names: the bulk endpoint does not accept them, so query one by one, in parallel, over 18-month windows
    from concurrent.futures import ThreadPoolExecutor
    wins, s0 = [], dt.date(2021, 1, 1)
    while s0 < SNAP["npm"]:
        e0 = min(s0 + dt.timedelta(days=540), SNAP["npm"] - dt.timedelta(days=1))
        wins.append((s0, e0)); s0 = e0 + dt.timedelta(days=1)

    def one(n):
        for s1, e1 in wins:
            j = get(f"https://api.npmjs.org/downloads/point/{s1}:{e1}/{quote(n, safe='@/')}")
            if j is None:
                return None
            if (j.get("downloads") or 0) > 0:
                return True
        return False
    with ThreadPoolExecutor(8) as ex:
        for i, (n, p) in enumerate(zip(scoped, ex.map(one, scoped)), 1):
            if p is not None:
                save([{"eco": "npm", "name": n, "prior": p}])
            if i % 500 == 0:
                print(f"  npm scoped {i}/{len(scoped)}", flush=True)


def main() -> None:
    h = mech.hallucinated(st.CUTOFF, "all")
    ctl = mech.controls(st.CUTOFF, "all", drop="none")
    allnames = pd.concat([h[["eco", "name"]], ctl[["eco", "name"]]]).drop_duplicates()
    done = load()
    todo = allnames[[(e, n) not in done for e, n in zip(allnames.eco, allnames.name)]]
    print(f"{len(allnames)} names, {len(todo)} to check", flush=True)
    import sys
    skip_scoped = "--skip-scoped" in sys.argv     # scoped names can only be registered by the scope owner, so a
    pypi(sorted(todo.name[todo.eco == "pypi"]))   # deleted scoped package cannot be revived by an outside attacker
    npm_todo = sorted(todo.name[todo.eco == "npm"])
    npm([n for n in npm_todo if not (skip_scoped and n.startswith("@"))])
    done = load()
    # npm: not used. The bulk download endpoint silently returns null for many packages that single-name queries show
    # were downloaded (verified on live packages from 2014-2017), and single-name queries are rate-limited. npm keeps a
    # tombstone with the creation time of every unpublished package, which 07_stats.status() already uses to exclude
    # names registered (and removed) before the cutoff, so download history is needed only for PyPI, which deletes
    # projects without trace.
    out = allnames.assign(prior_existence=[done.get((e, n)) if e == "pypi" else None
                                           for e, n in zip(allnames.eco, allnames.name)])
    out.to_parquet(DERIVED / "prior_existence.parquet", index=False)
    print(out.groupby("eco").prior_existence.agg(["count", "sum"]))


if __name__ == "__main__":
    main()
