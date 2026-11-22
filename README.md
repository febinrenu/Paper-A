# Hallucinated, then claimed?

A time-capsule measurement of LLM package hallucinations on PyPI and npm, 2024 → 2026.

Open code models released before 2024 cannot know packages registered after their training data was collected.
When such a model recommends a package that did not exist in early 2024 but exists today, someone registered
that name in the meantime. This project measures how often that happens, when, by whom, and whether the
same names are still recommended by current local models.

## Pipeline

| Step | Script | Output |
|---|---|---|
| Inputs | `src/00_fetch_inputs.py` | Prompt sets and early-2024 registry snapshots released by Spracklen et al. (USENIX Security 2025) |
| Registries | `src/01_snapshot_registries.py` | Current PyPI list, OSV malicious-package records, snapshot dating |
| Generation | `src/02_generate.py` | Code + package recommendations from local models (Ollama) |
| Extraction | `src/extract.py` | One deterministic parser for every model |

Further steps (classification, enrichment, controls, statistics, figures) are added as the study progresses.

## Reproducing

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt
python src/00_fetch_inputs.py
python src/01_snapshot_registries.py
python src/02_generate.py --models codellama:7b-instruct,qwen2.5-coder:7b
```

Hardware used: single RTX 4060 Laptop GPU (8 GB), Ollama with 4-bit quantised models.

## Data handling and ethics

- No package names are registered by this project.
- Raw generations and lists of still-unregistered hallucinated names are **not** published in clear text,
  because they could guide squatting. Released data contains aggregates, already-registered names, and salted hashes.
- Package contents are only downloaded as text for static inspection and are never installed or executed.
- Suspicious live packages are reported to the PyPI and npm security teams before publication.

## Credits

Prompt sets, early-2024 registry snapshots and false-positive lists are from
Spracklen et al., *We Have a Package for You!* (USENIX Security 2025),
[Spracks/PackageHallucination](https://github.com/Spracks/PackageHallucination), Zenodo 10.5281/zenodo.14676377.
