"""Blind validation sheets for the author (written to local/validation/, never published: contains package names).

1. claimant_sheet.csv (100 rows): claimed hallucinated packages, stratified by the automatic class, which is NOT
   shown. The labeller assigns: malicious / placeholder / spam / empty_squat / organic / unclear.
   Key: claimant_key.csv. Agreement (Cohen's kappa) is computed by 16_validation_sheets.py --score.
2. semantic_sheet.csv (60 rows): a programming prompt and a package description. 40 are real (the prompt
   produced the package name), 20 are shuffled. The labeller rates whether the package would serve the prompt:
   2 = clearly, 1 = partly, 0 = no. Key: semantic_key.csv. Scored as real-vs-shuffled mean rating.
Usage: python src/16_validation_sheets.py            (make sheets; refuses to overwrite filled ones)
       python src/16_validation_sheets.py --score    (after labelling: writes tables/u6_validation.csv)
"""
from __future__ import annotations

import argparse
import importlib
import json
import random

import numpy as np
import pandas as pd

from common import DERIVED, ROOT, TABLES

OUT = ROOT / "local" / "validation"
SEED = 1122
QUOTA = {"organic": 35, "unclear": 25, "malicious": 15, "empty_squat": 10, "placeholder": 10, "removed_unknown": 5}


def url(eco, name):
    return f"https://pypi.org/project/{name}/" if eco == "pypi" else f"https://www.npmjs.com/package/{name}"


def make() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for f in ("claimant_sheet.csv", "semantic_sheet.csv"):
        p = OUT / f
        if p.exists() and pd.read_csv(p).iloc[:, -2].notna().any():
            raise SystemExit(f"{p} already has labels; not overwriting")
    rng = random.Random(SEED)
    c = pd.read_parquet(DERIVED / "claimants.parquet")
    e = pd.read_parquet(DERIVED / "enriched.parquet")[["eco", "name", "summary", "description_head"]]
    c = c.merge(e, on=["eco", "name"], how="left")
    parts = []
    for cls, k in QUOTA.items():
        g = c[c["class"] == cls]
        parts.append(g.sample(min(k, len(g)), random_state=SEED))
    s = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    s["id"] = [f"C{i + 1:03d}" for i in range(len(s))]
    sheet = pd.DataFrame({"id": s.id, "eco": s.eco, "name": s.name, "link": [url(a, b) for a, b in zip(s.eco, s.name)],
                          "created": s.created, "releases": s.n_releases, "has_repo": s.has_repo,
                          "summary": s.summary.fillna("").str[:300], "description_start": s.description_head.fillna("").str[:400],
                          "your_label": "", "notes": ""})
    sheet.to_csv(OUT / "claimant_sheet.csv", index=False, encoding="utf-8-sig")
    s[["id", "eco", "name", "class"]].to_csv(OUT / "claimant_key.csv", index=False)

    mech = importlib.import_module("10_mechanism")
    st = importlib.import_module("07_stats")
    h = mech.hallucinated(st.CUTOFF)
    live = h[h.claimed & (h.status == "claimed_live")].copy()
    live["text"] = (live.summary.fillna("") + " " + live.description_head.fillna("")).str.strip().str[:500]
    live = live[live.text.str.len() > 30]
    mentions = pd.read_parquet(DERIVED / "mentions.parquet")
    m = mentions[mentions.era == 2024].merge(live[["eco", "name"]], on=["eco", "name"]).drop_duplicates(["eco", "name"])
    prompts = {}
    for lang in ("Python", "Javascript"):
        for sset in ("SO_AT", "SO_LY", "LLM_AT", "LLM_LY"):
            lines = [l for l in (mech.SPRACKLEN / "Data" / lang / f"{sset}.json").read_text(encoding="utf-8").splitlines() if l.strip()]
            for i, l in enumerate(lines):
                o = json.loads(l)
                prompts[(lang, sset, i)] = str(o if isinstance(o, str) else next(iter(o.values())))[:600]
    m["prompt"] = [prompts.get((a, b, c_)) for a, b, c_ in zip(m.lang, m.set, m.idx)]
    m = m[m.prompt.notna()].merge(live[["eco", "name", "text"]], on=["eco", "name"])
    pick = m.sample(min(60, len(m)), random_state=SEED).reset_index(drop=True)
    real, rest = pick.iloc[:40], pick.iloc[40:]
    shuf_text = list(rest.text)
    rng.shuffle(shuf_text)
    shuffled = pd.DataFrame({"prompt": list(rest.prompt), "text": shuf_text, "eco": list(rest.eco),
                             "name": ["(shuffled)"] * len(rest)})
    rows = pd.concat([real.assign(kind="real")[["prompt", "text", "eco", "name", "kind"]],
                      shuffled.assign(kind="shuffled")]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    rows["id"] = [f"S{i + 1:03d}" for i in range(len(rows))]
    rows[["id", "prompt", "text"]].rename(columns={"text": "package_description"}).assign(rating_0_1_2="", notes="") \
        .to_csv(OUT / "semantic_sheet.csv", index=False, encoding="utf-8-sig")
    rows[["id", "kind", "eco", "name"]].to_csv(OUT / "semantic_key.csv", index=False)
    print(f"wrote {len(sheet)} claimant rows and {len(rows)} semantic rows to {OUT}")


def kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = (a == b).mean()
    pe = sum((a == k).mean() * (b == k).mean() for k in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def score() -> None:
    rows = []
    cs, ck = pd.read_csv(OUT / "claimant_sheet.csv"), pd.read_csv(OUT / "claimant_key.csv")
    d = cs.merge(ck, on="id")
    d = d[d.your_label.notna() & (d.your_label.astype(str).str.strip() != "")]
    if len(d):
        lab = d.your_label.str.strip().str.lower()
        auto = d["class"].replace({"removed_unknown": "unclear"})
        rows.append({"check": "claimant class", "n": len(d), "agreement": (lab == auto).mean(), "kappa": kappa(lab, auto)})
        mal = (lab == "malicious") == (auto == "malicious")
        rows.append({"check": "malicious vs not", "n": len(d), "agreement": mal.mean(),
                     "kappa": kappa(lab == "malicious", auto == "malicious")})
    ss, sk = pd.read_csv(OUT / "semantic_sheet.csv"), pd.read_csv(OUT / "semantic_key.csv")
    d = ss.merge(sk, on="id")
    d = d[pd.to_numeric(d.rating_0_1_2, errors="coerce").notna()]
    if len(d):
        from scipy.stats import mannwhitneyu
        r = pd.to_numeric(d.rating_0_1_2)
        a, b = r[d.kind == "real"], r[d.kind == "shuffled"]
        rows.append({"check": "semantic real vs shuffled", "n": len(d), "real_mean": a.mean(), "shuffled_mean": b.mean(),
                     "real_share_clear": (a == 2).mean(), "p_one_sided": mannwhitneyu(a, b, alternative="greater").pvalue})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "u6_validation.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    score() if ap.parse_args().score else make()
