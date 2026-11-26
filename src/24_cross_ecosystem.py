"""A meaning-preserving baseline: the same hallucinated name, in the other registry.

Token-swap controls keep a name's shape but not its meaning. Here the comparison name is the hallucinated name
itself, looked up in the registry it was *not* recommended for: a name a model invented for PyPI, as an npm name, and
vice versa. Such a name is exactly as natural, plausible and on-topic as the hallucination, but no model recommended
it to users of that registry. If attackers simply favoured natural-sounding names, it should be targeted about as
often as the hallucination; if what matters is that a model recommends the name for that ecosystem, it should be
targeted about as rarely as a control.

Eligible cross names: absent from the target registry's 2024 snapshot and never emitted by any model for the target
registry. PyPI names are written as their PEP 503 normal form (a valid npm name); npm names are normalised the same
way for PyPI, and scoped npm names are skipped. Outcome: a post-cutoff, non-withdrawn OSV MAL record in the target
registry, exactly as for hallucinated names and controls.
Outputs: tables/u11_cross_ecosystem.csv (aggregates only).
"""
from __future__ import annotations

import importlib
import re

import numpy as np
import pandas as pd
from scipy import stats

from common import DERIVED, REGISTRY, TABLES

st = importlib.import_module("07_stats")
mech = importlib.import_module("10_mechanism")


def pep503(n: str) -> str:
    return re.sub(r"[-_.]+", "-", n).lower()


def main() -> None:
    cut = st.CUTOFF
    post = mech.osv_post(cut)
    h = mech.hallucinated(cut, "all")
    c = mech.controls(cut, "all")
    names = pd.read_parquet(DERIVED / "names.parquet")
    emitted = {e: set(names.name[names.eco == e]) for e in ("npm", "pypi")}
    snap = {"pypi": set((REGISTRY / "pypi_2024.txt").read_text(encoding="utf-8").split("\n")),
            "npm": set((REGISTRY / "npm_2024.txt").read_text(encoding="utf-8").split("\n"))}
    rows = []
    for src, dst in (("pypi", "npm"), ("npm", "pypi")):
        hs = h[h.eco == src]
        cand = hs.name if src == "pypi" else hs.name[~hs.name.str.startswith("@")]
        cross = {pep503(n) for n in cand}
        cross = {n for n in cross if n not in snap[dst] and n not in emitted[dst]}
        k = sum((dst, n) in post for n in cross)
        hd, cd = h[h.eco == dst], c[c.eco == dst]
        hk, ck = int(hd.mal_post.sum()), int(cd.mal_post.sum())
        f_c = stats.fisher_exact([[k, len(cross) - k], [ck, len(cd) - ck]], alternative="greater")[1]
        f_h = stats.fisher_exact([[hk, len(hd) - hk], [k, len(cross) - k]], alternative="greater")[1]
        rows.append({"names_from": src, "registry": dst, "cross_n": len(cross), "cross_mal": k,
                     "cross_rate": k / len(cross), "halluc_rate": hk / len(hd), "control_rate": ck / len(cd),
                     "rr_cross_vs_control": (k / len(cross)) / (ck / len(cd)) if ck else np.nan,
                     "rr_halluc_vs_cross": (hk / len(hd)) / (k / len(cross)) if k else np.nan,
                     "p_cross_gt_control": f_c, "p_halluc_gt_cross": f_h})
    out = pd.DataFrame(rows)
    # pooled
    tot = out[["cross_n", "cross_mal"]].sum()
    out.to_csv(TABLES / "u11_cross_ecosystem.csv", index=False)
    print(out.to_string(index=False))
    print("pooled cross rate", tot.cross_mal / tot.cross_n)


if __name__ == "__main__":
    main()
