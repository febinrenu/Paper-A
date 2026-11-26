"""Paper figures, built only from the aggregate CSVs in tables/ (vector PDF + PNG preview in figures/).

Palette: validated reference categorical palette (blue = hallucinated, orange = control; fixed order for
claimant classes). Single y-axis per chart, thin marks, recessive grid, direct labels where few series.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates  # noqa: E402,F401
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common import FIGURES, TABLES  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
CLASS_ORDER = ["malicious", "removed_unknown", "placeholder", "empty_squat", "unclear", "organic"]
CLASS_COLORS = dict(zip(CLASS_ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]))
CLASS_LABEL = {"malicious": "Malicious", "removed_unknown": "Removed (no report)", "placeholder": "Defensive placeholder",
               "empty_squat": "Empty squat", "unclear": "Unclear", "organic": "Real project"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
ECO = {"npm": "npm", "pypi": "PyPI"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.facecolor": SURF, "axes.facecolor": SURF, "pdf.fonttype": 42,
})


def read_csv(name):
    p = TABLES / name
    return pd.read_csv(p) if p.exists() else None


def save(fig, name):
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIGURES.parent / "paper" / f"{name}.pdf", bbox_inches="tight")    # manuscript folder is self-contained
    plt.close(fig)


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_malware_vs_controls():
    a = pd.read_csv(TABLES / "u1a_malware_vs_controls.csv")
    a = a[(a.cutoff == "2024-03-31") & a.eco.isin(["npm", "pypi"]) & (a.get("set", "all") == "all")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4))
    for ax, outcome, title in zip(axes, ["malicious", "claimed"], ["Later carried malware (OSV MAL)", "Later registered (any)"]):
        d = a[a.outcome == outcome].set_index("eco")
        y = np.arange(len(d))
        for i, eco in enumerate(d.index):
            r = d.loc[eco]
            for k, n, color, off, lab in ((r.halluc_k, r.halluc_n, BLUE, 0.12, "Hallucinated by 2024 models"),
                                          (r.control_k, r.control_n, ORANGE, -0.12, "Control names (token swaps)")):
                lo, hi = wilson(k, n)
                ax.plot([lo * 100, hi * 100], [i + off] * 2, color=color, lw=2, solid_capstyle="round")
                ax.plot(k / n * 100, i + off, "o", ms=6, color=color, mec=SURF, mew=1.5, label=lab if i == 0 else None)
            right = max(wilson(r.halluc_k, r.halluc_n)[1], wilson(r.control_k, r.control_n)[1]) * 100
            ax.text(right, i, f"  {r.risk_ratio:.1f}×", va="center", ha="left", color=INK, fontsize=8.5, fontweight="bold")
        ax.set_yticks(y, [ECO[e] for e in d.index])
        ax.set_xlabel("% of names (95% CI)")
        ax.set_title(title, color=INK, fontsize=9, loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, ax.get_xlim()[1] * 1.22)
        ax.set_ylim(-0.6, len(d) - 0.4)
    axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=2)
    save(fig, "fig_malware_vs_controls")


def fig_timeline():
    s = pd.read_csv(TABLES / "u1_monthly_claims.csv")
    s["rate"] = s.claims / s.n_names * 1000
    s["month_ts"] = pd.PeriodIndex(s.month, freq="M").to_timestamp()
    fig, ax = plt.subplots(figsize=(6.6, 2.4))
    for g, color, lab in (("hallucinated", BLUE, "Hallucinated by 2024 models"), ("control", ORANGE, "Control names (token swaps)")):
        d = s[s.group == g].sort_values("month_ts")
        ax.plot(d.month_ts, d.rate.rolling(3, min_periods=1).mean(), color=color, lw=2, label=lab)
    import importlib
    ms = importlib.import_module("07_stats").MILESTONES
    ax.set_ylim(0, ax.get_ylim()[1] * 1.05)
    short = {"Spracklen et al. arXiv": "Spracklen et al.\n(arXiv)", "'slopsquatting' coined": "'slopsquatting'\ncoined",
             "USENIX Security 2025": "USENIX\nSec '25"}
    for label, day in ms.items():
        t = pd.Timestamp(day)
        if t >= s.month_ts.min():
            ax.axvline(t, color=AXIS, lw=0.8, ls="--", zorder=0)
            ax.text(t, ax.get_ylim()[1], short.get(label, label), va="bottom", ha="center", fontsize=6.5,
                    color=MUTED, clip_on=False)
    ax.set_ylabel("Registrations per 1,000 names\n(3-month rolling mean)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2)
    save(fig, "fig_timeline")


def fig_probe_budget():
    pb = pd.read_csv(TABLES / "rq5_probe_budget.csv")
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]       # sequential blue, light -> dark
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), sharey=True)
    for ax, eco in zip(axes, ["npm", "pypi"]):
        d = pb[pb.eco == eco]
        for m in sorted(d.models.unique()):
            g = d[d.models == m].sort_values("prompts_per_model")
            col = ramp[min(int(m) - 1, len(ramp) - 1)]
            ax.plot(g.prompts_per_model, g.malicious_recall * 100, "-o", color=col, lw=2, ms=4.5, mec=SURF, mew=1)
            last = g.iloc[-1]
            ax.text(last.prompts_per_model * 1.12, last.malicious_recall * 100, f"{int(m)} model{'s' if m > 1 else ''}",
                    va="center", fontsize=7, color=INK2)
        ax.set_xscale("log")
        ax.set_xticks([100, 300, 1000, 3000], ["100", "300", "1k", "3k"])
        ax.minorticks_off()
        ax.set_xlim(80, 7000)
        ax.set_title(ECO[eco], color=INK, fontsize=9, loc="left")
        ax.set_xlabel("Prompts per model (log scale)")
        ax.set_ylim(0, 108)
    axes[0].set_ylabel("Later-malicious names\npre-empted (%)")
    fig.text(0.5, -0.12, "Each model × 3k prompts ≈ 3–5 GPU-hours on a laptop GPU (measured 3.3–5.9 s per prompt); averages over 50 random draws of models and prompts.",
             ha="center", fontsize=7, color=MUTED)
    save(fig, "fig_probe_budget")


def fig_claimants():
    c = pd.read_csv(TABLES / "u6_claimants.csv")
    fig, ax = plt.subplots(figsize=(6.6, 1.7))
    for i, eco in enumerate(["npm", "pypi"]):
        d = c[c.eco == eco].set_index("class").n.reindex(CLASS_ORDER).fillna(0)
        tot, left = d.sum(), 0.0
        for cls, n in d.items():
            if n == 0:
                continue
            w = n / tot * 100
            ax.barh(i, w - 0.4, left=left + 0.2, height=0.62, color=CLASS_COLORS[cls], label=CLASS_LABEL[cls] if i == 0 else None)
            if w >= 6:
                txt = "#0b0b0b" if cls == "empty_squat" else "#ffffff"
                ax.text(left + w / 2, i, f"{int(n)}", ha="center", va="center", fontsize=7.5, color=txt, fontweight="bold")
            left += w
    ax.set_yticks([0, 1], [f"npm (n={int(c[c.eco=='npm'].n.sum())})", f"PyPI (n={int(c[c.eco=='pypi'].n.sum())})"])
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of registered hallucinated names")
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=7.5)
    save(fig, "fig_claimants")


def fig_design():
    """Study design on one time axis: frozen 2024 models, registry snapshots, cutoff, observation window."""
    from common import ALL_MODELS
    T = lambda d: pd.Timestamp(d)
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.grid(False)
    ax.set_xlim(T("2023-06-01"), T("2027-01-31"))
    ax.set_ylim(0.1, 4.3)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    rel24 = sorted(T(v["release"]) for v in ALL_MODELS.values() if v.get("era") == 2024 and v.get("release"))
    rel26 = sorted(T(v["release"]) for v in ALL_MODELS.values()
                   if v.get("era") == 2026 and v.get("release") and len(str(v["release"])) > 4)
    # row 3: 2024-era models released (weights frozen)
    ax.scatter(rel24, [3.4] * len(rel24), s=28, color=BLUE, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.text(rel24[0], 3.85, "11 open models released (weights frozen)", color=INK2, fontsize=8, va="bottom")
    # row 2: snapshots and cutoff
    for d, lab, y in ((T("2024-01-15"), "npm snapshot ", 2.55), (T("2024-03-12"), "PyPI snapshot ", 2.15)):
        ax.plot([d, d], [0.2, 3.0], color=AXIS, lw=1, ls=(0, (2, 2)))
        ax.text(d, y, lab, color=INK2, fontsize=7.5, ha="right")
    cut = T("2024-03-31")
    ax.plot([cut, cut], [0.2, 3.0], color=INK, lw=1.2)
    ax.text(cut, 3.0, " cutoff", color=INK, fontsize=8, va="bottom")
    # row 1: observation window
    end = T("2026-11-23")
    ax.add_patch(plt.Rectangle((cut, 0.95), end - cut, 0.5, color=BLUE, alpha=0.12, lw=0))
    ax.text(cut + (end - cut) / 2, 1.2, "2.5-year window: who registers the names, and with what?", ha="center",
            va="center", color=INK, fontsize=8)
    ax.plot([end, end], [0.2, 3.0], color=INK, lw=1.2)
    ax.text(end, 3.0, " freeze", color=INK, fontsize=8, va="bottom")
    # row 0: later models
    ax.scatter(rel26, [0.45] * len(rel26), s=22, color=ORANGE, zorder=3, edgecolor=SURF, linewidth=1.2)
    ax.text(rel26[-1] + pd.Timedelta(days=30), 0.45, "later models: do they still recommend the names?",
            color=INK2, fontsize=7.5, va="center")
    ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    save(fig, "fig_design")


def fig_incidence():
    d = read_csv("u14_incidence.csv")
    if d is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True)
    for ax, (w, title) in zip(axes, (("all", "All eleven 2024 models"), ("c1", "Pre-registered set C1"))):
        for arm, color, lab in (("hallucinated", BLUE, "Hallucinated names"), ("control", ORANGE, "Control names")):
            g = d[(d["set"] == w) & (d.arm == arm)]
            x = pd.PeriodIndex(g.month, freq="M").to_timestamp()
            ax.fill_between(x, 100 * g.lo, 100 * g.hi, color=color, alpha=0.15, lw=0)
            ax.plot(x, 100 * g.cum_incidence, color=color, lw=2, label=lab)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
    axes[0].set_ylabel("Names with a malware report (%)")
    axes[0].legend(loc="upper left", fontsize=7.5)
    save(fig, "fig_incidence")


def fig_forest():
    rows = []
    def add(lab, df, sets, an, group):
        if df is None:
            return
        for w in sets:
            x = df[(df["set"] == w) & (df.analysis == an)]
            if len(x) and pd.notna(x["or"].iloc[0]):
                rows.append((group, lab, w, float(x["or"].iloc[0]), float(x.or_lo.iloc[0]), float(x.or_hi.iloc[0])))
    u10, u12, u14 = read_csv("u10_outcome_robustness.csv"), read_csv("u12_reviewer_checks.csv"), read_csv("u14_review3.csv")
    both = ("all", "c1")
    add("All names", u10, both, "primary", "Main")
    add("Verified post-cutoff malware only", u14, both, "verified_post_cutoff_only", "Outcome")
    add("Excluding research and probe packages", u10, both, "excl_probe_research", "Outcome")
    add("Excluding dependency-confusion fingerprints", u12, both, "excl_depconf_fingerprint", "Outcome")
    add("Reported before April 2025", u12, both, "reported_before_2025_04", "Outcome")
    add("Reported from April 2025", u12, both, "reported_from_2025_04", "Outcome")
    add("Excluding names declared in public code", u12, both, "excl_declared_in_public_code", "Names")
    add("Excluding names with earlier downloads", u10, both, "excl_prior_existence", "Names")
    add("Registrable names only", u12, both, "registrable_only", "Names")
    add("Matched sets", u10, both, "matched_sets", "Dependence")
    add("One event per report day", u10, both, "one_event_per_report_day", "Dependence")
    add("Cluster bootstrap: name families", u14, both, "cluster_bootstrap_family", "Dependence")
    add("C1 events unlinked to discovery campaigns", u12, ("c1",), "campaign_independent", "Dependence")
    add("Unrelated-lineage models only", u10, ("c1",), "lineage_unrelated_models", "Dependence")
    add("Natural-name controls (token bigrams)", u14, ("all",), "natural_name_controls_pooled", "Controls")
    add("Same names in the other registry", u12, ("cross",), "cross_pooled_all", "Negative control")
    if not rows:
        return
    n_labels = len({(g, lab) for g, lab, *_ in rows})
    fig, ax = plt.subplots(figsize=(7.2, 0.3 * n_labels + 0.9))
    ax.grid(axis="y", visible=False)
    y = 0
    ticks, labels = [], []
    last = None
    for g, lab, w, o, lo, hi in rows:
        key = (g, lab)
        if key != last:
            y -= 1
            ticks.append(y); labels.append(lab)
            last = key
        off = 0.18 if w == "c1" else (-0.18 if w == "all" else 0)
        color = {"all": BLUE, "c1": ORANGE}.get(w, MUTED)
        ax.plot([lo, hi], [y + off, y + off], color=color, lw=2, solid_capstyle="round")
        ax.scatter([o], [y + off], s=22, color=color, zorder=3, edgecolor=SURF, linewidth=1)
    ax.axvline(1, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_yticks(ticks); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_xlabel("Odds ratio, hallucinated vs control names (log scale; 95% CI)")
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], color=BLUE, lw=2, marker="o", label="All eleven 2024 models"),
                       Line2D([], [], color=ORANGE, lw=2, marker="o", label="Pre-registered set C1"),
                       Line2D([], [], color=MUTED, lw=2, marker="o", label="Negative control")],
              loc="upper left", fontsize=7, frameon=True, framealpha=0.95, edgecolor=SURF)
    save(fig, "fig_forest")


def main() -> None:
    for f in (fig_design, fig_malware_vs_controls, fig_timeline, fig_probe_budget, fig_claimants, fig_incidence, fig_forest):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as ex:
            print("skip", f.__name__, ex.filename)


if __name__ == "__main__":
    main()
