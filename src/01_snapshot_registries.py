"""Freeze registry state for the study.

Outputs (data/registry/):
  pypi_2024.txt / npm_2024.txt   normalised name sets from the Spracklen early-2024 snapshots
  pypi_now.txt                   full current PyPI project list (PEP 691 simple JSON)
  osv_mal.parquet                all OSV MAL-* (malicious package) records for PyPI + npm
  snapshot_dating.json           estimated capture date of the 2024 snapshots
"""
from __future__ import annotations

import io
import json
import random
import statistics
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd

from common import REGISTRY, SPRACKLEN, client, get_with_retry, sha256, update_manifest
from extract import normalize_py

OSV = "https://osv-vulnerabilities.storage.googleapis.com/{eco}/all.zip"


def load_2024() -> tuple[set, set]:
    py = pd.read_csv(SPRACKLEN / "Data/Python/pypi_package_names.csv", header=None, dtype=str, keep_default_na=False)[0]
    js = pd.read_csv(SPRACKLEN / "Data/Javascript/npm_package_names.csv", header=None, dtype=str, keep_default_na=False)[0]
    return {normalize_py(x) for x in py if x}, {x.strip().lower() for x in js if x.strip()}


def pypi_now(c) -> set:
    r = get_with_retry(c, "https://pypi.org/simple/", headers={"Accept": "application/vnd.pypi.simple.v1+json"})
    r.raise_for_status()
    return {normalize_py(p["name"]) for p in r.json()["projects"]}


def osv_mal(c) -> pd.DataFrame:
    rows = []
    for eco in ("PyPI", "npm"):
        r = get_with_retry(c, OSV.format(eco=eco))
        r.raise_for_status()
        (REGISTRY / f"osv_{eco}_all.zip").write_bytes(r.content)
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for n in z.namelist():
                if not n.startswith("MAL-"):
                    continue
                d = json.loads(z.read(n))
                for a in d.get("affected", []):
                    p = a.get("package", {})
                    name = p.get("name", "")
                    rows.append({
                        "ecosystem": eco, "id": d["id"],
                        "name": normalize_py(name) if eco == "PyPI" else name.lower(),
                        "published": d.get("published"), "modified": d.get("modified"),
                        "withdrawn": d.get("withdrawn"), "summary": (d.get("summary") or "")[:300],
                        "all_versions": not a.get("versions") and not a.get("ranges"),
                    })
    return pd.DataFrame(rows)


def pypi_first_upload(c, name: str) -> str | None:
    r = get_with_retry(c, f"https://pypi.org/pypi/{name}/json")
    if r.status_code != 200:
        return None
    times = [f["upload_time_iso_8601"] for files in r.json().get("releases", {}).values() for f in files]
    return min(times) if times else None


def npm_created(c, name: str) -> str | None:
    r = get_with_retry(c, f"https://registry.npmjs.org/{name.replace('/', '%2F')}", headers={"Accept": "application/json"})
    if r.status_code != 200:
        return None
    return r.json().get("time", {}).get("created")


def date_snapshots(c, py24: set, pynow: set, npm24: list) -> dict:
    """PyPI: creation dates of names present now but absent in the snapshot -> the earliest
    dense cluster marks the capture date. npm: max creation date over a random sample of
    snapshot names (lower bound)."""
    rng = random.Random(7)
    new = sorted(pynow - py24)
    sample = rng.sample(new, min(400, len(new)))
    created = sorted(t for t in (pypi_first_upload(c, n) for n in sample) if t)
    pct = lambda xs, q: xs[int(q * (len(xs) - 1))] if xs else None
    npm_sample = rng.sample(npm24, 400)
    npm_created_dates = sorted(t for t in (npm_created(c, n) for n in npm_sample) if t)
    return {
        "pypi_new_names_total": len(new),
        "pypi_new_created_p01": pct(created, 0.01), "pypi_new_created_p05": pct(created, 0.05),
        "pypi_new_created_p10": pct(created, 0.10), "pypi_new_created_median": pct(created, 0.5),
        "npm_sample_created_max": npm_created_dates[-1] if npm_created_dates else None,
        "npm_sample_created_p99": pct(npm_created_dates, 0.99),
        "note": "PyPI 'new' names include deleted-and-recreated projects, so use p05/p10, not min.",
    }


def main() -> None:
    t = datetime.now(timezone.utc).isoformat(timespec="seconds")
    py24, npm24 = load_2024()
    (REGISTRY / "pypi_2024.txt").write_text("\n".join(sorted(py24)), encoding="utf-8")
    (REGISTRY / "npm_2024.txt").write_text("\n".join(sorted(npm24)), encoding="utf-8")
    print("2024 snapshot sizes: pypi", len(py24), "npm", len(npm24))
    with client(timeout=300) as c:
        now = pypi_now(c)
        (REGISTRY / "pypi_now.txt").write_text("\n".join(sorted(now)), encoding="utf-8")
        print("pypi now:", len(now))
        mal = osv_mal(c)
        mal.to_parquet(REGISTRY / "osv_mal.parquet", index=False)
        print("OSV MAL records:", mal.groupby("ecosystem").size().to_dict())
        dating = date_snapshots(c, py24, now, sorted(npm24))
    dating["captured_at"] = t
    dating["pypi_2024_size"], dating["npm_2024_size"], dating["pypi_now_size"] = len(py24), len(npm24), len(now)
    (REGISTRY / "snapshot_dating.json").write_text(json.dumps(dating, indent=2))
    print(json.dumps(dating, indent=2))
    update_manifest({f"registry/{p.name}": sha256(p) for p in REGISTRY.iterdir() if p.suffix in {".txt", ".parquet"}}
                    | {"registry/captured_at": t})


if __name__ == "__main__":
    main()
