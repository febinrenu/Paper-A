# Pre-specified analysis plan (committed before the confirmatory data exist)

Written 2026-11-24 ~00:35 IST. The git commit timestamp of this file is the registration time.

## Already observed: the discovery sample

**Data at the time of writing:**
- Five 2024-era models: CodeLlama-7B, DeepSeek-Coder 6.7B and 1.3B, Mistral-7B v0.2 and OpenChat-3.5. OpenChat was 77% complete.
- 3,000 prompts each.
- Registry state looked up 2026-11-24.
- Cutoff 2024-03-31.

**Observed results:**
- Names hallucinated by these models (absent from the early-2024 snapshot, and not pre-cutoff registrations) later carried OSV-confirmed malware (`MAL-*` record published after the cutoff) at:
  - **npm:** 0.62% vs 0.086% for frequency-weighted token-swap controls (K=3), ratio 7.3, Fisher p ≈ 1e-9
  - **PyPI:** 0.22% vs 0.023%, ratio 9.7, p ≈ 7e-5
- Any registration: PyPI 2.3× controls (p ≈ 5e-13); npm 1.1×, not significant.

These numbers are exploratory. The hypotheses below are tested on data that did not exist when this file was committed.

## Confirmatory sample 1: six additional 2024-era models

Models not yet run at the time of writing:
- `llama2:7b-chat`
- `magicoder:7b`
- `deepseek-llm:7b-chat`
- `zephyr:7b`
- `qwen:7b-chat` (Qwen1.5)
- `gemma:7b-instruct-q4_0` (Gemma 1.0)

All use the same prompts, seeds, protocol and extractor.

**Confirmatory name set C1:** names emitted by at least one extension model, absent from the 2024 snapshot, not registered before the cutoff, and **not emitted by any discovery model**.

**Controls:** built by `src/06_controls.py` (same procedure, seed 1122) from C1.

| Hypothesis | Test | Decision rule |
|---|---|---|
| **H1 (primary)** | Share of C1 names with a post-cutoff OSV MAL record vs C1 controls | Pooled over ecosystems, one-sided Fisher exact (CMH stratified by ecosystem as secondary). Supported if OR > 1 and p < 0.05. Effect size with 95% CI reported regardless. |
| **H2** | Share of C1 names registered at all (claimed) vs controls | Same test. Expected direction > 1 on PyPI; no directional prediction for npm (spam-inflated baseline). |
| **H3 (dose-response)** | Within all 2024-cohort names, P(malicious) increases with the number of distinct 2024 models emitting the name | Logistic regression, `malicious ~ log1p(mentions) + n_models_2024 + C(eco)`. Supported if the n_models coefficient is > 0 with p < 0.05. |

## Confirmatory sample 2: the 2026 local cohort (not yet run)

Models:
- `qwen2.5-coder:7b`
- `qwen3:8b`
- `gemma3:4b`
- `llama3.2:3b`
- `phi4-mini`
- `qwen3.5:4b`

| Hypothesis | Prediction |
|---|---|
| **H4 (persistence)** | At least 1% of 2024-cohort hallucinated names that are still unregistered are emitted by at least one 2026 local model (Wilson 95% CI reported). |
| **H5 (live risk)** | Some names that carry malware now, or carried it after the cutoff, are still emitted by 2026 local or hosted models. Reported as counts per model; no significance test. |
| **H6 (temporal placebo)** | Names hallucinated *only* by 2026 models (absent from 2024 and never emitted by any 2024-era model) have a lower post-cutoff malicious rate than 2024-cohort names. Fisher exact, one-sided. |

## Fixed analysis choices

| Choice | Setting |
|---|---|
| Primary cutoff | 2024-03-31; sensitivity at 2024-02-29 and 2024-04-30 |
| Name normalisation and filters | `src/extract.py` as of this commit |
| Exclusions | Names that only ever appear as dotted module paths are reported separately |
| npm | Results reported with and without spam and placeholder claims once `src/12_claimants.py` exists. The spam rule must be fixed before it is applied to C1. |
| Multiple testing | H1 is the single primary test. The others are secondary and reported with unadjusted p-values and CIs. |
| Deviations | Any deviation from this plan is listed in the paper as a deviation |
