"""Same generation protocol as 02_generate.py, for the large hosted cohort (NVIDIA API, OpenAI-compatible).

Identical prompts, prompt sample, seeds, system instructions and sampling parameters; only the transport
differs. Reasoning is switched off via chat_template_kwargs (the selected models honour it; models that
ignore it were excluded). Each API key gets its own request-rate limiter; models are spread over keys.
Output rows go to data/gen/<model>.jsonl with the same schema plus "backend" and "served_model".

Usage: python src/02b_generate_hosted.py [--models nim:a,nim:b] [--rpm 36] [--concurrency 4]
Keys are read from .env (NVIDIA_API_KEY, NVIDIA_API_KEY_2, ...); never logged.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import random
import time

import httpx

from common import GEN, HOSTED, ROOT

gen = importlib.import_module("02_generate")


def load_env() -> dict:
    p = ROOT / ".env"
    return dict(l.split("=", 1) for l in p.read_text(encoding="utf-8").splitlines() if "=" in l and not l.startswith("#"))


class RateLimiter:
    """Evenly spaced requests at `rpm` per minute; slows down after HTTP 429."""

    def __init__(self, rpm: float):
        self.base = self.interval = 60.0 / rpm
        self.next_t = 0.0
        self.lock = asyncio.Lock()

    def ok(self):
        """Successful request: drift back towards the configured rate."""
        self.interval = max(self.base, self.interval * 0.97)

    async def wait(self):
        async with self.lock:
            now = time.monotonic()
            t = max(now, self.next_t)
            self.next_t = t + self.interval
        await asyncio.sleep(max(0.0, t - time.monotonic()))

    def penalise(self, seconds: float):
        self.next_t = max(self.next_t, time.monotonic() + seconds)
        self.interval = min(self.interval * 1.25, 60.0)


class KeyPool:
    """All keys shared by all models: each request goes to the key whose rate limiter frees up first."""

    def __init__(self, keys: list[str], rpm: float):
        self.keys = keys
        self.limiters = [RateLimiter(rpm) for _ in keys]

    def pick(self) -> int:
        return min(range(len(self.keys)), key=lambda i: self.limiters[i].next_t)


def make_chat(base: str, pool: KeyPool, stats: dict):
    async def nim_chat(c: httpx.AsyncClient, model: str, messages: list[dict], opts: dict, think) -> dict:
        body = {"model": model.removeprefix("nim:"), "messages": messages, "stream": False,
                "temperature": opts["temperature"], "top_p": opts["top_p"], "top_k": opts["top_k"],
                "max_tokens": opts["num_predict"], "seed": opts["seed"],
                "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}
        last = ""
        for attempt in range(10):
            ki = pool.pick()
            key, limiter = pool.keys[ki], pool.limiters[ki]
            await limiter.wait()
            try:
                r = await c.post(f"{base}/chat/completions", json=body, headers={"Authorization": f"Bearer {key}"})
                stats["requests"] += 1
                if r.status_code == 429:
                    stats["429"] += 1
                    if stats["429"] <= 3:
                        print(f"  429 detail: {r.text[:200]} retry-after={r.headers.get('retry-after')}", flush=True)
                    wait = float(r.headers.get("retry-after") or 30)
                    limiter.penalise(wait)
                    last = "HTTP 429"
                    continue
                if r.status_code != 200:
                    last = f"HTTP {r.status_code}: {r.text[:300]}"
                    raise httpx.HTTPStatusError(last, request=r.request, response=r)
                limiter.ok()
                j = r.json()
                ch = j["choices"][0]
                msg = ch["message"]
                return {"text": msg.get("content") or "", "eval_count": (j.get("usage") or {}).get("completion_tokens"),
                        "done_reason": ch.get("finish_reason"), "eval_ms": None, "resampled": 0,
                        "served_model": j.get("model"), "reasoning_chars": len(msg.get("reasoning_content") or "")}
            except (httpx.HTTPError, KeyError, ValueError, IndexError) as ex:
                last = last or f"{type(ex).__name__}: {ex}"[:300]
                await asyncio.sleep(min(10 * (attempt + 1), 120) + random.random() * 5)
        raise RuntimeError(f"chat failed after retries: {last}")

    return nim_chat


async def run_model(model: str, rows: list[dict], chat_fn, concurrency: int, stats: dict) -> None:
    out = gen.model_file(model)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["lang"], r["set"], r["idx"]))
    gave_up = {k for k, v in gen.failure_counts(model).items() if v >= gen.MAX_FAILS}
    todo = [r for r in rows if (r["lang"], r["set"], r["idx"]) not in done | gave_up]
    print(f"[{model}] {len(done)} done, {len(gave_up)} given up, {len(todo)} to go", flush=True)
    if not todo:
        return
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    t0, n = time.time(), 0
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as c:

        async def worker(row):
            nonlocal n
            async with sem:
                try:
                    res = await gen.run_one(c, model, row, None, gen.SAMPLE_SEED, chat_fn=chat_fn)
                except Exception as ex:  # noqa: BLE001
                    async with lock:
                        with open(gen.ERRORS, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"model": model, "lang": row["lang"], "set": row["set"], "idx": row["idx"],
                                                "error": f"{type(ex).__name__}: {ex}"[:500],
                                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                    print(f"[{model}] prompt {row['lang']}/{row['set']}/{row['idx']} failed: {str(ex)[:150]}", flush=True)
                    return
            res["backend"] = "nvidia-api"
            async with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n += 1
                if n % 25 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"[{model}] {n}/{len(todo)}  {rate:.1f}s/prompt  eta {(len(todo)-n)*rate/3600:.1f}h  "
                          f"(requests {stats['requests']}, 429s {stats['429']})", flush=True)

        await asyncio.gather(*(worker(r) for r in todo), return_exceptions=True)


# run_one stores only text/tokens/done_reason; keep the served model id too
_orig_run_one = gen.run_one


async def _run_one_with_meta(c, model, row, think, seed_base, chat_fn=None):
    seen = {}

    async def spy(*a, **k):
        res = await chat_fn(*a, **k)
        seen.setdefault("served_model", res.get("served_model"))
        seen["reasoning_chars"] = seen.get("reasoning_chars", 0) + (res.get("reasoning_chars") or 0)
        return res

    row_out = await _orig_run_one(c, model, row, think, seed_base, chat_fn=spy)
    row_out.update(seen)
    return row_out


gen.run_one = _run_one_with_meta


async def main_async(models: list[str], rpm: float, concurrency: int, per_set: int, limit: int) -> None:
    env = load_env()
    base = env.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
    keys = [env[k].strip() for k in sorted(env) if k.startswith("NVIDIA_API_KEY")]
    rows = gen.load_prompts(per_set)
    if limit:
        rows = [r for lang in gen.LANGS for s in gen.SETS
                for r in [x for x in rows if x["lang"] == lang and x["set"] == s][:limit]]
    pool = KeyPool(keys, rpm)
    jobs = []
    for m in models:
        st = {"requests": 0, "429": 0}           # per-model counters
        jobs.append(run_model(m, rows, make_chat(base, pool, st), concurrency, st))
    print(time.strftime("%Y-%m-%d %H:%M"), f"hosted run: {len(models)} models sharing {len(keys)} keys at {rpm} rpm/key", flush=True)
    await asyncio.gather(*jobs)
    print(time.strftime("%Y-%m-%d %H:%M"), "hosted run finished", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(HOSTED))
    ap.add_argument("--rpm", type=float, default=36)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--per-set", type=int, default=375)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    asyncio.run(main_async([m.strip() for m in a.models.split(",")], a.rpm, a.concurrency, a.per_set, a.limit))


if __name__ == "__main__":
    main()
