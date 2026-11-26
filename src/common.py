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
    "qwen2.5-coder:7b":                 {"era": 2026, "family": "Qwen2.5-Coder", "release": "2024-09-19"},
    "qwen3:8b":                         {"era": 2026, "family": "Qwen3", "release": "2025-04-29"},
    "gemma3:4b":                        {"era": 2026, "family": "Gemma3", "release": "2025-03-12"},
    "llama3.2:3b":                      {"era": 2026, "family": "Llama3.2", "release": "2024-09-25"},
    "phi4-mini":                        {"era": 2026, "family": "Phi-4-mini", "release": "2025-02-26"},
    "qwen3.5:4b":                       {"era": 2026, "family": "Qwen3.5", "release": "2026"},  # exact date verified before writing
    # 2024-era extension (added 2026-11-24; run after the 2026 local cohort). Tags verified by weight digest:
    # gemma:7b-instruct now points to Gemma 1.1 (Apr 2024) -> the older q4_0 tag (Gemma 1.0) is used instead;
    # qwen:7b-chat == qwen:7b-chat-v1.5 (Qwen1.5); magicoder:7b == 7b-s-cl; zephyr:7b == 7b-beta.
    "llama2:7b-chat":                   {"era": 2024, "family": "Llama2", "release": "2023-07-18"},
    "magicoder:7b":                     {"era": 2024, "family": "Magicoder", "release": "2023-12-04"},
    "deepseek-llm:7b-chat":             {"era": 2024, "family": "DeepSeek-LLM", "release": "2023-11-29"},
    "zephyr:7b":                        {"era": 2024, "family": "Zephyr", "release": "2023-10-25"},
    "qwen:7b-chat":                     {"era": 2024, "family": "Qwen1.5", "release": "2024-02-05"},
    "gemma:7b-instruct-q4_0":           {"era": 2024, "family": "Gemma1", "release": "2024-02-21"},
}
# discovery = the five models whose results were seen before docs/analysis_plan.md was committed;
# extension = the six confirmatory models named in that plan (set C1 / hypothesis H1)
DISCOVERY = ("codellama:7b-instruct", "deepseek-coder:6.7b-instruct", "deepseek-coder:1.3b-instruct",
             "mistral:7b-instruct-v0.2-q4_K_M", "openchat:7b-v3.5")
EXTENSION = ("llama2:7b-chat", "magicoder:7b", "deepseek-llm:7b-chat", "zephyr:7b", "qwen:7b-chat", "gemma:7b-instruct-q4_0")
for _k, _m in MODELS.items():
    _m["cohort"] = f"{_m['era']}-local"
    _m["role"] = "discovery" if _k in DISCOVERY else "extension" if _k in EXTENSION else None

# 2026 large hosted cohort (NVIDIA API, full precision). Kept separate from MODELS because the local
# driver only runs models present in Ollama. Release dates verified before writing.
HOSTED = {
    "nim:nvidia/nemotron-3.5-lightning-30b-a3b": {"era": 2026, "cohort": "2026-hosted", "family": "Nemotron-3.5", "release": None},
    "nim:nvidia/nemotron-3-ultra-550b-a55b":     {"era": 2026, "cohort": "2026-hosted", "family": "Nemotron-3", "release": None},
    "nim:minimaxai/minimax-m3.1-flash":          {"era": 2026, "cohort": "2026-hosted", "family": "MiniMax-M3.1", "release": None},
    "nim:poolside/laguna-xs-2.1":                {"era": 2026, "cohort": "2026-hosted", "family": "Laguna", "release": None},
}
ALL_MODELS = {**MODELS, **HOSTED}

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
