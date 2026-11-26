"""Do public GitHub projects declare a malicious hallucinated package as a dependency?

For each hallucinated name (2024-era cohort) with a post-cutoff, non-withdrawn OSV MAL record, GitHub code search
(read-only, authenticated through the gh CLI, 10 requests/min) for the name in package.json (npm) or
requirements*.txt / pyproject.toml (PyPI). Every hit is verified by reading the file and checking that the name is
declared as a dependency (exact name; PyPI names normalised), not merely mentioned. Up to 100 hits are verified
per query; total_count is reported as an upper bound.

Outputs: data/derived/manifests.parquet (name-level, not published), tables/u2c_manifests.csv (aggregates only).
No repository is named in any output. Owners of repositories that still declare a malicious package are notified
separately.
"""
from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd

from common import DERIVED, TABLES

# GitHub client of the companion census (rate-limit aware, pauses while offline); loaded under its own name
_spec = importlib.util.spec_from_file_location(
    "census_common", Path(__file__).resolve().parents[2] / "agent-config-census" / "src" / "common.py")
_cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_cc)
GitHub = _cc.GitHub

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")
CACHE = DERIVED / "manifests_cache.jsonl"
DEP_KEYS = ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies")


def norm(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def declares(eco: str, name: str, path: str, text: str) -> bool:
    if eco == "npm":
        try:
            j = json.loads(text)
        except ValueError:
            return False
        return isinstance(j, dict) and any(isinstance(j.get(k), dict) and name in j[k] for k in DEP_KEYS)
    target = norm(name)
    pat = re.compile(r"""^\s*["']?([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(?:[<>=!~;@,"'\s]|$)""")
    for line in text.splitlines():
        line = line.split("#", 1)[0]
        if path.endswith(".toml"):
            for m in re.finditer(r"""["']([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(?:[<>=!~;@ ][^"']*)?["']""", line):
                if norm(m.group(1)) == target:
                    return True
            m = re.match(r"""^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*=""", line)     # poetry: name = "^1.0"
            if m and norm(m.group(1)) == target:
                return True
        else:
            m = pat.match(line)
            if m and norm(m.group(1)) == target:
                return True
    return False


def load_cache() -> dict:
    out = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            out[(d["eco"], d["name"], d["query"])] = d
    return out


def search(gh: GitHub, gh_core: GitHub, eco: str, name: str, query: str) -> dict:
    r = gh.get("/search/code", params={"q": f'"{name}" {query}', "per_page": 100})
    if r is None or r.status_code != 200:
        return {"eco": eco, "name": name, "query": query, "total": None, "checked": 0, "verified": [], "error": True}
    j = r.json()
    verified, checked = [], 0
    for it in j.get("items", []):
        f = gh_core.get(it["url"])
        if f is None or f.status_code != 200:
            continue
        checked += 1
        try:
            text = base64.b64decode(f.json().get("content", "")).decode("utf-8", "replace")
        except (ValueError, KeyError):
            continue
        if declares(eco, name, it["path"], text):
            verified.append(it["repository"]["full_name"])
    return {"eco": eco, "name": name, "query": query, "total": j.get("total_count"), "checked": checked,
            "verified": sorted(set(verified)), "error": False}


def main() -> None:
    cand, _ = st.load_candidates(st.CUTOFF)
    post = mech.osv_post(st.CUTOFF)
    h = cand[(cand.n_models_2024 > 0) & (cand.status != "pre_cutoff")]
    h = h[[(e, n) in post for e, n in zip(h.eco, h.name)]]
    queries = {"npm": ["filename:package.json"], "pypi": ["filename:requirements.txt", "filename:pyproject.toml"]}
    cache = load_cache()
    gh, gh_core = GitHub(min_interval=6.5), GitHub()
    todo = [(r.eco, r.name, q) for r in h.itertuples() for q in queries[r.eco] if (r.eco, r.name, q) not in cache]
    print(f"{len(h)} malicious hallucinated names, {len(todo)} searches to run", flush=True)
    for i, (eco, name, q) in enumerate(todo, 1):
        res = search(gh, gh_core, eco, name, q)
        with open(CACHE, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        if i % 10 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
    cache = load_cache()
    rows = []
    for r in h.itertuples():
        res = [cache.get((r.eco, r.name, q)) for q in queries[r.eco]]
        res = [x for x in res if x]
        repos = sorted({x for d in res for x in d["verified"]})
        rows.append({"eco": r.eco, "name": r.name, "hits_total": sum((d["total"] or 0) for d in res),
                     "checked": sum(d["checked"] for d in res), "repos_declaring": len(repos),
                     "errors": sum(bool(d["error"]) for d in res), "repos": json.dumps(repos)})
    d = pd.DataFrame(rows)
    d.to_parquet(DERIVED / "manifests.parquet", index=False)
    out = d.groupby("eco").agg(names=("name", "size"), names_with_declaring_repo=("repos_declaring", lambda s: int((s > 0).sum())),
                               repos_declaring=("repos_declaring", "sum"), max_repos_one_name=("repos_declaring", "max"),
                               search_hits=("hits_total", "sum"), files_checked=("checked", "sum"),
                               search_errors=("errors", "sum")).reset_index()
    out.to_csv(TABLES / "u2c_manifests.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
