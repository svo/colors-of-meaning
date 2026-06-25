# Feature: Scaled, multi-dataset evaluation of the color method (honest numbers beyond 400 samples)

## Overview

The headline "Color Method" result in the README (AG News, **81.25% / F1 81.22%**)
is produced at `max_samples: 400` — 100 documents per class — because the color
histogram classifier re-ranks candidates with **exact** Earth-Mover distance
(`WassersteinDistanceCalculator._exact_earth_mover_distance`
`src/colors_of_meaning/infrastructure/ml/wasserstein_distance_calculator.py:44`,
`ot.emd2` over the 4096×4096 perceptual cost matrix, ~92 ms/call). A full AG News
test set (7,600 docs × up to 100 re-ranked candidates) is therefore hours of
compute, so the project's central retrieval/classification claim currently rests
on a tiny budget on a **single** dataset.

This feature makes the evaluation **honest at scale and across datasets**. It adds
a fast, **metric** histogram distance — sliced-Wasserstein over the fixed codebook
Lab support (`ot.sliced_wasserstein_distance`, established with the POT docs) — and
a **fidelity gate** that proves the fast distance ranks documents the same way
exact EMD does *before* any scaled number is trusted. With that proxy validated,
it runs the color method on the **full** (or a justified large) AG News test set
and on the two already-supported-but-unreported datasets, **IMDB** (binary
sentiment) and **20 Newsgroups** (20-class topic), and commits the resulting
numbers and the exact commands that reproduce them.

Nothing about the science changes silently: the sliced-Wasserstein distance is a
true lower-bound metric on the same perceptual Lab support, the candidate
retrieval already uses an `hnswlib` cosine index over histograms
(`ColorHistogramClassifier`
`src/colors_of_meaning/infrastructure/evaluation/color_histogram_classifier.py:41`),
and the re-rank simply swaps `emd2` for the sliced metric. The fidelity gate
reports the Spearman rank-correlation between the proxy and exact EMD on a held-out
subsample and the downstream-accuracy delta, so the speed-up is **measured, not
assumed**. No new third-party dependency is introduced: `pot`, `scikit-learn`,
`scipy`, and `hnswlib` are already declared (`setup.cfg` `install_requires`).

## Core Domain Concepts

- **Histogram distance proxy**: a fast `DistanceCalculator` over color histograms
  that approximates exact perceptual Wasserstein. Here, **sliced-Wasserstein** on
  the fixed 4096-color Lab support with the two documents' histograms as weights.
  It is a genuine metric on the same support, not a heuristic.
- **Proxy fidelity**: the agreement between the proxy and exact EMD, measured as
  (a) Spearman correlation of the two distance vectors over sampled document pairs
  and (b) the absolute accuracy delta of the full classifier under each. A proxy
  is only used at scale once its fidelity clears a documented threshold.
- **Evaluation budget**: `max_samples` per split (null = full split). The result
  is annotated with the budget so a 400-sample number is never confused with a
  full-test number.
- **Committed result table**: a checked-in `reports/eval_results.md` with one row
  per (dataset, method, distance, budget) carrying accuracy, macro-F1, MRR,
  recall@k, bits/token, and wall-clock — the auditable evidence behind the README.

## User Stories

- As a researcher, I want the color method's accuracy on the **full AG News test
  set**, not 400 samples, so the headline number is credible.
- As a researcher, I want committed **IMDB** and **20 Newsgroups** results so the
  method is shown to generalize beyond one topic dataset.
- As a skeptic, I want proof that the fast distance used to make scale feasible
  **ranks documents the same way exact EMD does**, so the speed-up did not quietly
  change the result.
- As a maintainer, I want the scaled runs to be **deterministic** and reproducible
  from a committed config and a single command.
- As a contributor, I want the fast distance behind the existing `DistanceCalculator`
  port so it drops into the current classifier with no pipeline rewrite.

## Acceptance Criteria

- [ ] Given the color method on AG News, when evaluated on the **full test split**
  (or a documented ≥4,000-sample budget if full is infeasible on CI hardware),
  then accuracy and macro-F1 are reported and committed, with the budget recorded.
- [ ] Given the sliced-Wasserstein proxy, when its fidelity is measured against
  exact EMD on a held-out subsample, then the Spearman rank-correlation of the
  distances is **≥ 0.95** and the downstream accuracy delta is **≤ 1.0 point**;
  below that threshold the run **fails loudly** rather than reporting the number.
- [ ] Given IMDB and given 20 Newsgroups, when the color method is evaluated, then
  accuracy and macro-F1 are reported and committed for each.
- [ ] Given the same dataset, distance, budget, and seed, when the evaluation runs
  twice, then the reported metrics are **identical**.
- [ ] Given a chosen distance method (`wasserstein` exact, `sliced`, `sinkhorn`,
  `jensen_shannon`), when `eval` runs, then it uses that distance and records which
  one produced the number.
- [ ] Given the committed `reports/eval_results.md`, when the README table is
  regenerated, then every cell traces to a row there with the command that
  produced it.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays 100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No new business rules. The fast distance implements the existing
`DistanceCalculator` ABC (`domain/service/distance_calculator.py`); no domain
model changes. The proxy-fidelity concept is expressed as a small immutable result
(either reuse `EvaluationResult` for the scaled metrics, or a new
`domain/model/distance_fidelity.py` dataclass carrying `spearman`,
`accuracy_delta`, `pair_count`, `is_faithful`). Domain stays free of `ot`/`scipy`
(architecture test enforces it, mirroring the existing scipy/sklearn isolation
rules in `tests/colors_of_meaning/test_synesthetic_architecture.py`).

### Application Layer (`src/colors_of_meaning/application/`)

`EvaluateUseCase` (`application/use_case/evaluate_use_case.py`) is reused
unchanged. New `EvaluateDistanceFidelityUseCase` encodes a sample of documents
once, computes proxy vs exact distances over sampled pairs, and returns the
fidelity result — the honesty gate. A thin `EvaluationSuiteUseCase` (optional)
orchestrates "fidelity-gate then evaluate" so the CLI stays declarative. Both
depend only on injected domain ports + the encode use case; `correlation-id`
logging added.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/ml/sliced_wasserstein_distance_calculator.py` →
`SlicedWassersteinDistanceCalculator(DistanceCalculator)`: builds the fixed Lab
support coordinates from the codebook once (as `WassersteinDistanceCalculator`
already does, `:54`) and computes `ot.sliced_wasserstein_distance(support,
support, hist1, hist2, n_projections, seed)`. Deterministic via a fixed seed.
`metric_name()` → `"sliced_wasserstein"`. No change to `ColorHistogramClassifier`
beyond receiving this calculator via its existing `distance_calculator` parameter.

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/cli/eval.py` gains `--distance {wasserstein,sliced,sinkhorn,jensen_shannon}`
and a `--max-samples` override (falling back to the config). New
`interface/cli/eval_suite.py` runs the fidelity gate then evaluates a list of
(dataset, method, distance) cells and writes/refreshes `reports/eval_results.md`.
New `[testenv:eval_suite]`. New committed configs `configs/agnews_full.yaml`,
`configs/imdb_run.yaml`, `configs/newsgroups_run.yaml`. README "Current
Performance" table extended to all three datasets with the distance and budget
columns and a pointer to `reports/eval_results.md`.

### Shared Layer

No changes.

## API Contracts

No API contract changes. Evaluation is an offline/CLI concern; the existing
`POST /query/palette` contract is unaffected.

## CLI Impact

- `eval`: add `--distance` (default `wasserstein` to preserve current behaviour)
  and `--max-samples`. No existing default behaviour changes.
- `eval_suite` (new): `--datasets`, `--distance`, `--fidelity-samples`,
  `--output-path` (default `reports/eval_results.md`). Runs the fidelity gate,
  then each cell, prints a table, and writes the committed report. Refuses to
  report a scaled number when the proxy fails the fidelity threshold.

## Dependency Injection

The CLIs construct the chosen `DistanceCalculator` and inject it into
`ColorHistogramClassifier`, exactly as `eval.py` already constructs
`WassersteinDistanceCalculator` (`:81`). The fidelity and suite use cases receive
their ports via constructors. No Lagom container or API wiring changes.

## Observability

`correlation-id` structured logging: the fidelity use case logs `{spearman,
accuracy_delta, pair_count, proxy, is_faithful}`; the suite logs `{dataset,
method, distance, budget, accuracy, macro_f1, seconds}` per cell. No new
metrics/tracing.

## Open Questions

- **Skip the re-rank entirely?** A fixed-projection sliced-Wasserstein *embedding*
  would let the `hnswlib` index rank directly, removing the per-pair re-rank.
  Default: keep the re-rank with the sliced metric (simpler, already fast enough);
  the embedding is a future optimization.
- **Full test set vs large subsample on CI.** Default: run the largest budget that
  fits the CI time box and record it; document the budget rather than silently
  sampling.
- **Sinkhorn as the proxy instead of sliced.** Sinkhorn is already wired
  (`sinkhorn_reg`) but is still per-pair and entropy-blurred. Default: sliced for
  scale; expose both and let the fidelity gate decide per dataset.
- **Embedding cache.** Re-embedding every doc per run dominates wall-clock.
  Default: in-run cache; a persistent embedding cache is a future option.
