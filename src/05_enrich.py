"""Registry metadata for every candidate name (absent from the 2024 snapshot).

PyPI: JSON API.  npm: registry document (includes unpublish tombstones and npm "security
holding" placeholders that replace removed malware). OSV MAL matches are joined from
data/registry/osv_mal.parquet. Responses are cached in data/derived/registry_cache.jsonl
so the step is resumable; author e-mail addresses are not stored.

Output: data/derived/enriched.parquet
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from urllib.parse import quote

import httpx
import pandas as pd

from common import DERIVED, REGISTRY, USER_AGENT

CACHE = DERIVED / "registry_cache.jsonl"


def _pypi_summary(j: dict) -> dict:
    info = j.get("info", {})
    files = [f for fs in j.get("releases", {}).values() for f in fs]
    times = sorted(f["upload_time_iso_8601"] for f in files)
    urls = info.get("project_urls") or {}
    return {
        "exists": True, "created": times[0] if times else None, "last_upload": times[-1] if times else None,
        "n_releases": len(j.get("releases", {})), "n_files": len(files),
        "total_size": sum(f.get("size", 0) for f in files),
        "summary": (info.get("summary") or "")[:500], "description_len": len(info.get("description") or ""),
        "description_head": (info.get("description") or "")[:1500],
        "author": (info.get("author") or "")[:100], "has_repo": any("github" in (u or "") or "gitlab" in (u or "")
                                                                    for u in [info.get("home_page"), *urls.values()]),
        "requires_dist": len(info.get("requires_dist") or []), "yanked_all": bool(files) and all(f.get("yanked") for f in files),
        "latest_version": info.get("version"),
        "latest_files": [{"url": f["url"], "type": f["packagetype"], "size": f["size"]} for f in j.get("urls", [])][:4],
    }


def _npm_summary(j: dict) -> dict:
    t = j.get("time", {}) or {}
    if "versions" not in j:
        return {"exists": False, "tombstone": True, "created": t.get("created"),
                "unpublished": (t.get("unpublished") or {}).get("time")}
    latest = (j.get("dist-tags") or {}).get("latest")
    v = (j.get("versions") or {}).get(latest, {}) if latest else {}
    scripts = v.get("scripts") or {}
    repo = j.get("repository") or v.get("repository") or {}
    repo_url = repo.get("url", "") if isinstance(repo, dict) else str(repo)
    desc = (j.get("description") or v.get("description") or "")
    return {
        "exists": True, "created": t.get("created"), "last_upload": t.get("modified"),
        "n_releases": len(j.get("versions") or {}), "latest_version": latest,
        "summary": desc[:500], "description_len": len(j.get("readme") or ""), "description_head": (j.get("readme") or "")[:1500],
        "maintainers": len(j.get("maintainers") or []), "has_repo": bool(repo_url),
        "install_scripts": sorted(k for k in scripts if k in {"preinstall", "install", "postinstall", "prepare"}),
        "install_script_text": " ; ".join(f"{k}: {scripts[k]}"[:300] for k in scripts if k in {"preinstall", "install", "postinstall"}),
        "dependencies": len(v.get("dependencies") or {}), "unpacked_size": (v.get("dist") or {}).get("unpackedSize"),
        "tarball": (v.get("dist") or {}).get("tarball"),
        "security_holding": "security holding package" in desc.lower() or (latest or "").endswith("-security"),
    }


async def fetch(c: httpx.AsyncClient, eco: str, name: str) -> dict:
    url = f"https://pypi.org/pypi/{name}/json" if eco == "pypi" else f"https://registry.npmjs.org/{quote(name, safe='@')}"
    last = ""
    for attempt in range(5):
        try:
            r = await c.get(url)
            if r.status_code == 404:
                return {"eco": eco, "name": name, "exists": False}
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retry", request=r.request, response=r)
            r.raise_for_status()
            j = r.json()
            s = _pypi_summary(j) if eco == "pypi" else _npm_summary(j)
            return {"eco": eco, "name": name, **s}
        except (httpx.HTTPError, json.JSONDecodeError, KeyError, TypeError) as ex:
            last = f"{type(ex).__name__}: {ex}"[:200]
            await asyncio.sleep(2 ** attempt)
    return {"eco": eco, "name": name, "error": True, "error_reason": last}


async def run(cands: list[tuple[str, str]]) -> None:
    done = set()
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if not d.get("error"):
                done.add((d["eco"], d["name"]))
    todo = [x for x in cands if x not in done]
    print(f"{len(done)} cached, {len(todo)} to fetch", flush=True)
    sem = {"pypi": asyncio.Semaphore(8), "npm": asyncio.Semaphore(16)}
    async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                                 timeout=60, follow_redirects=True) as c:
        n = 0

        async def one(eco, name):
            nonlocal n
            async with sem[eco]:
                d = await fetch(c, eco, name)
            with open(CACHE, "a", encoding="utf-8") as f:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
            n += 1
            if n % 1000 == 0:
                print(f"  {n}/{len(todo)}", flush=True)

        await asyncio.gather(*(one(e, nm) for e, nm in todo))


def unresolved(cands: list[tuple[str, str]]) -> set:
    """Names whose latest cache record is a lookup error (network), i.e. not known to exist or not exist."""
    last = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            last[(d["eco"], d["name"])] = bool(d.get("error"))
    return {x for x in cands if last.get(x, True)}


def wait_online(max_minutes: int = 180) -> bool:
    for i in range(max_minutes):
        try:
            httpx.get("https://pypi.org/simple/", timeout=20)
            httpx.get("https://registry.npmjs.org/", timeout=20)
            return True
        except httpx.HTTPError:
            if i == 0:
                print(time.strftime("%H:%M"), "registries unreachable; waiting for the connection", flush=True)
            time.sleep(60)
    return False


def run_until_clean(cands: list[tuple[str, str]], rounds: int = 6) -> int:
    """Look up every name; names that failed on the network are retried (after the connection is back) so a
    network error is never left standing in for a registry answer. Returns the number still unresolved."""
    for r in range(rounds):
        wait_online()
        asyncio.run(run(cands))
        left = unresolved(cands)
        if not left:
            return 0
        print(f"round {r + 1}: {len(left)} lookups failed on the network; retrying", flush=True)
        time.sleep(60)
    return len(unresolved(cands))


def main() -> None:
    cands = [tuple(l.split("\t")) for l in (DERIVED / "candidates.txt").read_text(encoding="utf-8").splitlines() if l]
    n_left = run_until_clean(cands)
    recs = {}
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        recs[(d["eco"], d["name"])] = d          # last write wins
    df = pd.DataFrame(recs.values())
    mal = pd.read_parquet(REGISTRY / "osv_mal.parquet")
    mal["eco"] = mal.ecosystem.map({"PyPI": "pypi", "npm": "npm"})
    agg = mal.groupby(["eco", "name"]).agg(mal_ids=("id", lambda s: ",".join(sorted(set(s)))),
                                           mal_first_published=("published", "min")).reset_index()
    df = df.merge(agg, on=["eco", "name"], how="left")
    df["osv_mal"] = df.mal_ids.notna()
    for col in ("latest_files", "install_scripts"):
        if col in df:
            df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, list) else x)
    df.to_parquet(DERIVED / "enriched.parquet", index=False)
    print(df.groupby("eco").agg(n=("name", "size"), exists=("exists", "sum"), osv_mal=("osv_mal", "sum")))
    if n_left:
        print(f"{n_left} lookups still unresolved (network); exit 1 so the pipeline does not treat the data as final")
        sys.exit(1)


if __name__ == "__main__":
    main()
