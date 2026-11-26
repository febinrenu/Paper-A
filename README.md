# Hallucinated package names attract malware

Code, aggregate results and documentation for the study *Hallucinated package names attract malware: a
time-capsule study of slopsquatting on PyPI and npm* (F. K. Renu, J. S. George, G. N. Sundar; Karunya Institute
of Technology and Sciences).

Open code models released before early 2024 cannot know about packages registered later. When such a model
recommends a package that was absent from an early-2024 registry snapshot and exists today, someone registered
that name in the meantime. Running eleven such models on 3,000 programming prompts each, we followed 26,820
hallucinated PyPI and npm names to November 2026 alongside control names of the same shape. Hallucinated names
later carried a package reported as malicious in OSV about nine times as often as controls (pre-registered
replication on held-out names: odds ratio 6.5), and the association survives a wide range of robustness checks.

## Contents

| Path | What it holds |
|---|---|
| `src/` | The complete pipeline, numbered in execution order (below) |
| `docs/analysis_plan.md` | The pre-registered analysis plan (commit `849e843`, 24 Nov 2026) |
| `docs/run_notes.md` | Generation settings, model-tag verification and every interruption, with timestamps |
| `tables/` | Every aggregate result reported in the paper (CSV); no package names |
| `figures/` | The paper's figures (PDF and PNG), built from `tables/` |
| `release/` | Hashed watchlist of still-unregistered hallucinated names (see `release/WATCHLIST.md`) |

## Pipeline

| Step | Script | Purpose |
|---|---|---|
| 0 | `00_fetch_inputs.py` | Prompt sets and early-2024 registry snapshots of Spracklen et al. |
| 1 | `01_snapshot_registries.py` | Current PyPI list, OSV `MAL-` records, snapshot dating |
| 2 | `02_generate.py`, `02b_generate_hosted.py` | Code and package recommendations from local (Ollama) and hosted models |
| 4 | `04_classify.py`, `extract.py` | Deterministic name extraction; hallucinated set |
| 5 | `05_enrich.py` | Registry metadata and OSV join for every name |
| 6 | `06_controls.py` | Token-swap control names |
| 7 | `07_stats.py` | Claim rates, persistence |
| 8 | `08_figures.py` | Figures |
| 9 | `09_defense.py` | Watchlist coverage and probe budget |
| 10 | `10_mechanism.py` | Malware vs controls, dose-response, campaigns, timing, semantic match |
| 11 | `11_exposure.py` | Downloads and dependents |
| 12 | `12_claimants.py` | Static classification of claimed packages (never installed or executed) |
| 13 | `13_confounds.py` | Plausibility and typo-proximity |
| 14 | `14_confirmatory.py` | Pre-registered hypotheses H1-H6 |
| 15 | `15_numbers.py` | Every number in the manuscript, as LaTeX macros |
| 16 | `16_validation_sheets.py` | Blind validation sheets and agreement |
| 17 | `17_detection.py` | Installable windows of malicious npm versions |
| 18 | `18_manifests.py` | Dependency declarations in public GitHub manifests |
| 19-20 | `19_settings_check.py`, `20_settings_rerun.py` | Inference-settings robustness |
| 21 | `21_watchlist.py` | Hashed watchlist |
| 22 | `22_prior_existence.py` | Names with download history before the snapshot |
| 23-27 | `23_outcome_robustness.py` ... `27_review3_checks.py` | Robustness and sensitivity analyses |
| 28 | `28_appendix.py` | Model digests and pipeline counts |

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python src/00_fetch_inputs.py
python src/01_snapshot_registries.py
python src/02_generate.py --models codellama:7b-instruct,qwen2.5-coder:7b   # see docs/run_notes.md for all models
python src/04_classify.py && python src/05_enrich.py && python src/06_controls.py --set discovery
# then the analysis scripts in numerical order
```

Hardware: one RTX 4060 Laptop GPU (8 GB), Ollama 0.40.1, 4-bit quantised models; exact weight digests are listed
in the paper's appendix and in `tables/u15_models.csv`. Registry state is time-dependent, so rerunning the
enrichment today will give a later snapshot than ours (frozen 22-26 November 2026).

## Data handling and ethics

- No package names were registered by this project.
- Raw generations and lists of still-unregistered hallucinated names are **not** published in clear text, because
  they could guide squatting. This repository contains code, aggregates and salted hashes only. The clear-text lists
  were given to the PyPI and npm security teams.
- Package contents were downloaded only as archives for static reading and never installed or executed.
- Researchers who need name-level data for replication can contact the corresponding author
  (febink@karunya.edu.in).

## Credits

Prompt sets, early-2024 registry snapshots and false-positive lists are from Spracklen et al., *We Have a Package
for You!* (USENIX Security 2025), [Spracks/PackageHallucination](https://github.com/Spracks/PackageHallucination),
Zenodo 10.5281/zenodo.14676377.
