"""Robustness: did the Ollama server-setting change (2026-11-23 15:15-15:18, docs/run_notes.md) move the estimates?

DeepSeek-Coder-6.7B was generated partly before and partly after the change, on disjoint prompts. Prompts were
processed in a fixed order, so the two halves also differ in prompt mix; we therefore compare within strata of
(language, prompt set) and report a Mantel-Haenszel odds ratio for "a mention is hallucinated" (absent from the
2024 snapshot) together with the raw shares, names per prompt and the share of prompts with any name.
Outputs: tables/u8_settings_check.csv.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import StratifiedTable

from common import DERIVED, GEN, TABLES

MODEL = "deepseek-coder:6.7b-instruct"
SWITCH_BEFORE, SWITCH_AFTER = pd.Timestamp("2026-11-23 15:15"), pd.Timestamp("2026-11-23 15:18")


def main() -> None:
    rows = []
    with open(GEN / "deepseek-coder__6.7b-instruct.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            rows.append({"lang": d["lang"], "set": d["set"], "idx": d["idx"], "ts": d.get("ts")})
    g = pd.DataFrame(rows)
    g["ts"] = pd.to_datetime(g.ts, errors="coerce", format="ISO8601")
    if g.ts.dt.tz is not None:
        g["ts"] = g.ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    g["period"] = np.where(g.ts < SWITCH_BEFORE, "before", np.where(g.ts >= SWITCH_AFTER, "after", "during"))
    g = g[g.period != "during"].drop_duplicates(["lang", "set", "idx"], keep="last")

    m = pd.read_parquet(DERIVED / "mentions.parquet")
    m = m[m.model == MODEL]
    names = pd.read_parquet(DERIVED / "names.parquet")[["eco", "name", "in_2024"]]
    m = m.merge(names, on=["eco", "name"], how="left").merge(g, on=["lang", "set", "idx"], how="inner")
    m["hall"] = ~m.in_2024.fillna(True).astype(bool)

    out = []
    for p, gp in g.groupby("period"):
        mp = m[m.period == p]
        per_prompt = mp.groupby(["lang", "set", "idx"]).size().reindex(
            pd.MultiIndex.from_frame(gp[["lang", "set", "idx"]]), fill_value=0)
        out.append({"period": p, "prompts": len(gp), "mentions": len(mp), "hallucinated_share": mp.hall.mean(),
                    "names_per_prompt": per_prompt.mean(), "share_prompts_with_name": (per_prompt > 0).mean()})
    tabs = []
    for _, s in m.groupby(["lang", "set"]):
        a, b = s[s.period == "after"], s[s.period == "before"]
        if len(a) and len(b):
            tabs.append(np.array([[a.hall.sum(), (~a.hall).sum()], [b.hall.sum(), (~b.hall).sum()]]) + 0.5)
    if tabs:
        st = StratifiedTable(tabs)
        lo, hi = st.oddsratio_pooled_confint()
        out.append({"period": "after_vs_before_MH", "or": st.oddsratio_pooled, "or_lo": lo, "or_hi": hi,
                    "p": st.test_null_odds().pvalue, "strata": len(tabs)})
    o = pd.DataFrame(out)
    o.to_csv(TABLES / "u8_settings_check.csv", index=False)
    print(o.to_string(index=False))


if __name__ == "__main__":
    main()
