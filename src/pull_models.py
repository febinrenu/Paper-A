"""Pull every cohort model into Ollama, retrying through network failures (partial blobs resume)."""
from __future__ import annotations

import json
import sys
import time
import urllib.request

from common import MODELS

OLLAMA = "http://127.0.0.1:11434"


def available() -> set:
    with urllib.request.urlopen(f"{OLLAMA}/api/tags") as r:
        names = {m["name"] for m in json.load(r)["models"]}
    return names | {n.removesuffix(":latest") for n in names}


def pull(model: str) -> bool:
    req = urllib.request.Request(f"{OLLAMA}/api/pull", data=json.dumps({"name": model, "stream": False}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=7200) as r:
            res = json.load(r)
        return res.get("status") == "success"
    except Exception as ex:  # noqa: BLE001
        print(f"  {model}: {str(ex)[:150]}", flush=True)
        return False


def main() -> None:
    order = sys.argv[1].split(",") if len(sys.argv) > 1 else list(MODELS)
    for attempt in range(1, 41):
        missing = [m for m in order if m not in available()]
        if not missing:
            print("all models present", flush=True)
            return
        print(time.strftime("%H:%M"), f"attempt {attempt}, missing: {missing}", flush=True)
        for m in missing:
            ok = pull(m)
            print(time.strftime("%H:%M"), m, "ok" if ok else "failed", flush=True)
        time.sleep(120)
    print("giving up after 40 rounds", flush=True)


if __name__ == "__main__":
    main()
