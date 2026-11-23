"""Generate code + package recommendations with local models via Ollama.

Protocol mirrors Spracklen et al. (USENIX Sec '25):
  step 'code': system instruction + prompt -> code      (T=0.7, top_k=20, top_p=0.9)
  step 't1'  : "Which <lang> packages are required to run this code: <code>"   (T=0.01, 64 tok)
  step 't2'  : "What <lang> packages would be useful in solving ...: <prompt>"  (T=0.01, 64 tok)
Deviation (documented): code generation capped at 1024 new tokens (theirs 2048) for throughput.

Resumable: rows already present in data/gen/<model>.jsonl are skipped.
Usage: python src/02_generate.py --models m1,m2 [--per-set 375] [--limit N] [--concurrency 4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
import zlib
from pathlib import Path

import httpx

from common import GEN, SPRACKLEN

OLLAMA = "http://127.0.0.1:11434"
LANGS = {"Python": "Python", "Javascript": "JavaScript"}
SETS = ["SO_AT", "SO_LY", "LLM_AT", "LLM_LY"]
SAMPLE_SEED = 20261122
MAX_PROMPT_CHARS = 4000

SYS_CODE = ("You are a coding assistant that generates {lang} code. Provide only the {lang} code and add additional "
            "explanatory text only when absolutely necessary. If no code is required to answer the question, simply reply 'None'")
SYS_T1 = ("You are a coding assistant that determines {lang} packages necessary to execute code. Respond with only a list of "
          "{lang} packages, separated by commas and no additional text or formatting. Your response must begin with the name of a {lang} package.")
PRE_T1 = "Which {lang} packages are required to run this code: "
SYS_T2 = ("You are a coding assistant that recommends {lang} packages that would be helpful to solve given problems. Respond with only "
          "a list of {lang} packages, separated by commas and no additional text or formatting. Your response must begin with the name of a {lang} package.")
PRE_T2 = "What {lang} packages would be useful in solving the following coding problem: "


def load_prompts(per_set: int) -> list[dict]:
    rows = []
    for lang_dir in LANGS:
        for s in SETS:
            path = SPRACKLEN / "Data" / lang_dir / f"{s}.json"
            prompts = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                text = obj if isinstance(obj, str) else next(iter(obj.values()))
                prompts.append(str(text))
            idx = list(range(len(prompts)))
            random.Random(f"{SAMPLE_SEED}-{lang_dir}-{s}").shuffle(idx)
            for i in sorted(idx[:per_set]):
                rows.append({"lang": lang_dir, "set": s, "idx": i, "prompt": prompts[i][:MAX_PROMPT_CHARS]})
    return rows


ERRORS = GEN / "errors.jsonl"
MAX_FAILS = 3   # a prompt that fails this many separate times is recorded as failed and skipped


def model_file(model: str) -> Path:
    return GEN / (model.replace(":", "__").replace("/", "_") + ".jsonl")


def failure_counts(model: str) -> dict:
    counts: dict = {}
    if ERRORS.exists():
        for line in ERRORS.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if e["model"] == model:
                k = (e["lang"], e["set"], e["idx"])
                counts[k] = counts.get(k, 0) + 1
    return counts


PLACEMENT = {"num_gpu": 99}   # all layers on the GPU (Ollama's estimate is conservative); affects speed only

# Thermal pacing (laptop): stop starting new prompts at HOT degC, resume at COOL degC. Speed only.
HOT, COOL = 83, 76


def gpu_temp() -> int | None:
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class Thermal:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.paused_s = 0.0
        self.last_check = 0.0

    async def gate(self, model: str):
        async with self.lock:                      # one coroutine checks; others queue behind it
            if time.time() - self.last_check < 5:
                return
            self.last_check = time.time()
            t = await asyncio.to_thread(gpu_temp)
            if t is None or t < HOT:
                return
            t0 = time.time()
            print(time.strftime("%H:%M"), f"[{model}] GPU {t} C: cooling to {COOL} C", flush=True)
            while t is not None and t > COOL:
                await asyncio.sleep(10)
                t = await asyncio.to_thread(gpu_temp)
            self.paused_s += time.time() - t0


async def chat(c: httpx.AsyncClient, model: str, messages: list[dict], opts: dict, think: bool | None) -> dict:
    body = {"model": model, "messages": messages, "stream": False, "options": {**opts, **PLACEMENT}, "keep_alive": "30m"}
    if think is not None:
        body["think"] = think
    last, resampled = "", 0
    for attempt in range(6):
        try:
            r = await c.post(f"{OLLAMA}/api/chat", json=body)
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:300]}"
                if "repeat limit" in r.text:
                    # degenerate repetition loop aborted by the runtime: resample with a new seed
                    # (documented in the paper; counted per model as 'resampled')
                    body["options"] = {**body["options"], "seed": (body["options"].get("seed", 0) + 7919 * (attempt + 1)) % (2**31)}
                    resampled += 1
                raise httpx.HTTPStatusError(last, request=r.request, response=r)
            j = r.json()
            return {"text": j["message"]["content"], "eval_count": j.get("eval_count"),
                    "done_reason": j.get("done_reason"), "eval_ms": (j.get("eval_duration") or 0) / 1e6,
                    "resampled": resampled}
        except (httpx.HTTPError, KeyError, ValueError) as ex:
            last = last or f"{type(ex).__name__}: {ex}"[:300]
            await asyncio.sleep(5 * (attempt + 1))
    raise RuntimeError(f"chat failed after retries: {last}")


async def run_one(c, model, row, think, seed_base, chat_fn=None):
    """One prompt through the 3-step protocol. chat_fn lets other backends (hosted API) reuse it unchanged."""
    chat_fn = chat_fn or chat
    lang = LANGS[row["lang"]]
    seed = (seed_base + zlib.crc32(f'{row["lang"]}|{row["set"]}|{row["idx"]}'.encode())) % (2**31)
    code = await chat_fn(c, model, [{"role": "user", "content": SYS_CODE.format(lang=lang) + row["prompt"]}],
                         {"temperature": 0.7, "top_k": 20, "top_p": 0.9, "num_predict": 1024, "seed": seed}, think)
    low = {"temperature": 0.01, "top_k": 20, "top_p": 0.9, "num_predict": 64, "seed": seed}
    t1 = await chat_fn(c, model, [{"role": "system", "content": SYS_T1.format(lang=lang)},
                                  {"role": "user", "content": PRE_T1.format(lang=lang) + code["text"].strip()}], low, think)
    t2 = await chat_fn(c, model, [{"role": "system", "content": SYS_T2.format(lang=lang)},
                                  {"role": "user", "content": PRE_T2.format(lang=lang) + row["prompt"].strip()}], low, think)
    return {"model": model, "lang": row["lang"], "set": row["set"], "idx": row["idx"],
            "code": code["text"], "t1": t1["text"], "t2": t2["text"],
            "tokens": {"code": code["eval_count"], "t1": t1["eval_count"], "t2": t2["eval_count"]},
            "code_done": code["done_reason"],
            "resampled": {"code": code["resampled"], "t1": t1["resampled"], "t2": t2["resampled"]},
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}


async def run_model(model: str, rows: list[dict], concurrency: int) -> None:
    out = model_file(model)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["lang"], r["set"], r["idx"]))
    fails = failure_counts(model)
    gave_up = {k for k, v in fails.items() if v >= MAX_FAILS}
    todo = [r for r in rows if (r["lang"], r["set"], r["idx"]) not in done | gave_up]
    think = False if model.startswith(("qwen3", "deepseek-r1", "qwen3.5")) else None
    print(f"[{model}] {len(done)} done, {len(gave_up)} given up, {len(todo)} to go", flush=True)
    if not todo:
        return
    async with httpx.AsyncClient(timeout=httpx.Timeout(600.0)) as c:
        info = (await c.post(f"{OLLAMA}/api/show", json={"model": model})).json()
        meta = GEN / "models.json"
        m = json.loads(meta.read_text()) if meta.exists() else {}
        m[model] = {"details": info.get("details"), "modified_at": info.get("modified_at")}
        meta.write_text(json.dumps(m, indent=2))

        sem = asyncio.Semaphore(concurrency)
        lock = asyncio.Lock()
        thermal = Thermal()
        t0, n = time.time(), 0

        async def worker(row):
            nonlocal n
            async with sem:
                await thermal.gate(model)
                try:
                    res = await run_one(c, model, row, think, SAMPLE_SEED)
                except Exception as ex:  # noqa: BLE001  -- one bad prompt must not stop the run
                    async with lock:
                        with open(ERRORS, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"model": model, "lang": row["lang"], "set": row["set"], "idx": row["idx"],
                                                "error": f"{type(ex).__name__}: {ex}"[:500],
                                                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
                    print(f"[{model}] prompt {row['lang']}/{row['set']}/{row['idx']} failed: {str(ex)[:150]}", flush=True)
                    return
            async with lock:
                with open(out, "a", encoding="utf-8") as f:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                n += 1
                if n % 25 == 0 or n == len(todo):
                    rate = (time.time() - t0) / n
                    print(f"[{model}] {n}/{len(todo)}  {rate:.1f}s/prompt  eta {(len(todo)-n)*rate/3600:.1f}h  "
                          f"(cooling pauses {thermal.paused_s/60:.0f} min)", flush=True)

        await asyncio.gather(*(worker(r) for r in todo), return_exceptions=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--per-set", type=int, default=375)
    ap.add_argument("--limit", type=int, default=0, help="pilot: first N prompts per language/prompt-set")
    ap.add_argument("--concurrency", type=int, default=4)
    a = ap.parse_args()
    rows = load_prompts(a.per_set)
    if a.limit:
        rows = [r for lang in LANGS for s in SETS
                for r in [x for x in rows if x["lang"] == lang and x["set"] == s][: a.limit]]
    for model in a.models.split(","):
        asyncio.run(run_model(model.strip(), rows, a.concurrency))


if __name__ == "__main__":
    main()
