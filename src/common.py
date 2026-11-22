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

# Study cohort. "release" = public release of the weights; training data predates it.
MODELS = {
    # 2024-era cohort (same families as Spracklen et al.; released before the early-2024 registry snapshot)
    "codellama:7b-instruct":            {"era": 2024, "family": "CodeLlama", "release": "2023-08-24"},
    "deepseek-coder:6.7b-instruct":     {"era": 2024, "family": "DeepSeek-Coder", "release": "2023-11-02"},
    "deepseek-coder:1.3b-instruct":     {"era": 2024, "family": "DeepSeek-Coder", "release": "2023-11-02"},
    "mistral:7b-instruct-v0.2-q4_K_M":  {"era": 2024, "family": "Mistral", "release": "2023-12-11"},
    "openchat:7b-v3.5":                 {"era": 2024, "family": "OpenChat", "release": "2023-11-01"},
    # 2026-era cohort (current small local models)
    "qwen2.5-coder:7b":                 {"era": 2026, "family": "Qwen2.5-Coder", "release": "2024-11-12"},
    "qwen3:8b":                         {"era": 2026, "family": "Qwen3", "release": "2025-04-29"},
    "gemma3:4b":                        {"era": 2026, "family": "Gemma3", "release": "2025-03-12"},
    "llama3.2:3b":                      {"era": 2026, "family": "Llama3.2", "release": "2024-09-25"},
    "phi4-mini":                        {"era": 2026, "family": "Phi-4-mini", "release": "2025-02-26"},
    "qwen3.5:4b":                       {"era": 2026, "family": "Qwen3.5", "release": "2026"},  # exact date verified before writing
}

USER_AGENT ="pkg-hallucination-study/0.1 (academic measurement; contact via repository)"


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
