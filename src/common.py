"""Shared paths, HTTP client and helpers."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SPRACKLEN = RAW / "spracklen"
REGISTRY = DATA / "registry"
GEN = DATA / "gen"
DERIVED = DATA / "derived"
QUARANTINE = DATA / "quarantine"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
MANIFEST = DATA / "MANIFEST.json"

for _p in (RAW, SPRACKLEN, REGISTRY, GEN, DERIVED, QUARANTINE, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

USER_AGENT = "pkg-hallucination-study/0.1 (academic measurement; contact via repository)"


def client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(
        headers={"User-Agent": USER_AGENT},
        timeout=timeout,
        follow_redirects=True,
        http2=False,
    )


def get_with_retry(c: httpx.Client, url: str, *, retries: int = 5, **kw) -> httpx.Response:
    delay = 2.0
    for attempt in range(retries):
        try:
            r = c.get(url, **kw)
            if r.status_code in (429, 500, 502, 503, 504):
                raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
            return r
        except (httpx.TransportError, httpx.HTTPStatusError):
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def update_manifest(entries: dict) -> None:
    m = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    m.update(entries)
    MANIFEST.write_text(json.dumps(m, indent=2, sort_keys=True))
