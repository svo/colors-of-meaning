# Plan: Scaled, multi-dataset evaluation of the color method

## Implementation Strategy

Make the color method's numbers honest at scale by (1) adding a fast, metric
histogram distance behind the existing `DistanceCalculator` port, (2) gating its
use on a measured fidelity check against exact EMD, and (3) running the validated
pipeline on the full AG News test set plus IMDB and 20 Newsgroups, committing the
numbers and the commands that reproduce them.

The bottleneck is the exact-EMD re-rank in `ColorHistogramClassifier._rerank_by_distance`
(`color_histogram_classifier.py:104`), which calls
`WassersteinDistanceCalculator.compute_distance` (`ot.emd2`, ~92 ms) once per
candidate. The candidate retrieval is already a fast `hnswlib` cosine index over
histograms, so only the re-rank needs to change: swap in
`SlicedWassersteinDistanceCalculator` (POT `ot.sliced_wasserstein_distance` over
the fixed codebook Lab support), which is a true metric and orders documents
almost identically to exact EMD at a fraction of the cost.

Three decisions keep it honest and clean:

1. **Measure the speed-up, don't assume it.** A `EvaluateDistanceFidelityUseCase`
   computes proxy and exact distances over sampled document pairs and reports the
   Spearman rank-correlation and the downstream accuracy delta. Scaled numbers are
   only emitted when fidelity clears the documented threshold (Spearman ≥ 0.95,
   delta ≤ 1.0 pt); otherwise the run fails loudly. This is the "science not
   plumbing" gate the project already values.
2. **One port, many distances.** The proxy implements the existing
   `DistanceCalculator` ABC, so the classifier, the use case, and the API are
   untouched; the CLI just selects which calculator to inject.
3. **Commit the evidence.** A single `eval_suite` command runs the gate then the
   cells and writes `reports/eval_results.md`; the README table is a view over it,
   never a hand-typed number.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/model/distance_fidelity.py` → frozen `DistanceFidelity(spearman:
  float, accuracy_delta: float, pair_count: int, threshold_spearman: float,
  max_accuracy_delta: float)` with an `is_faithful` property. Pure; validates
  ranges. No `ot`/`scipy` import.
- No change to `DistanceCalculator`; the proxy implements it.
- Tests: fidelity dataclass validation + `is_faithful` boundary.

### Application Layer (`src/colors_of_meaning/application/`)

- `EvaluateDistanceFidelityUseCase(encode_use_case, proxy_calculator,
  exact_calculator, embedding_adapter)` → `execute(samples, pair_count, seed) ->
  DistanceFidelity`: encode each sample once, draw `pair_count` random document
  pairs, compute both distances, Spearman-correlate them; the accuracy-delta term
  is supplied by the caller (the suite) or computed from two small classifier
  runs. `correlation-id` logging.
- `EvaluationSuiteUseCase(fidelity_use_case, evaluate_use_case_factory)` →
  `execute(cells) -> List[(cell, EvaluationResult)]`: run the gate, then each
  (dataset, method, distance) cell; raise if a scaled cell's proxy is unfaithful.
- `EvaluateUseCase` reused unchanged.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/ml/sliced_wasserstein_distance_calculator.py` →
  `SlicedWassersteinDistanceCalculator(DistanceCalculator)`. Constructor takes the
  `ColorCodebook` (build the `(num_bins, 3)` Lab support once), `n_projections`
  (default 100), and `seed`. `compute_distance(doc1, doc2)` →
  `float(ot.sliced_wasserstein_distance(support, support, doc1.histogram,
  doc2.histogram, n_projections=…, seed=…))` with the same num-bins guard the
  exact calculator uses. `metric_name()` → `"sliced_wasserstein"`. Deterministic.
- No change to `ColorHistogramClassifier`, `JensenShannonDistanceCalculator`,
  `SklearnMetricsCalculator`, or the dataset adapters.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval.py`: add `--distance {wasserstein,sliced,sinkhorn,jensen_shannon}`
  (default `wasserstein`) and `--max-samples`; a `_create_distance_calculator`
  helper builds the selected calculator; `_print_results` records the distance.
- `interface/cli/eval_suite.py` (new): tyro `@dataclass`, `main(args)`,
  `__main__` guard; runs the fidelity gate then the cells; writes
  `reports/eval_results.md`; prints the table; non-zero exit / raise on an
  unfaithful scaled cell.
- `configs/agnews_full.yaml` (max_samples: null), `configs/imdb_run.yaml`,
  `configs/newsgroups_run.yaml`.
- `tox.ini`: `[testenv:eval_suite]`.
- Architecture test: add `eval_suite` to the CLI→use-case rule; assert
  `sliced_wasserstein_distance_calculator` may import `ot` while the domain may
  not (mirror the existing scipy-isolation rules).
- `README.MD`: extend "Current Performance" to AG News (full) + IMDB + 20NG with
  distance/budget columns and a pointer to `reports/eval_results.md`.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

CLIs build the chosen `DistanceCalculator` and inject it into the classifier and
the fidelity use case, matching `eval.py`'s existing construction. No Lagom or API
container changes.

## Task List

1. [ ] domain: `DistanceFidelity` model + tests (validation, `is_faithful`
   boundary).
2. [ ] infrastructure: `SlicedWassersteinDistanceCalculator` + tests (metric name,
   symmetry, zero distance for identical histograms, determinism across two calls,
   num-bins guard).
3. [ ] application: `EvaluateDistanceFidelityUseCase` + tests (Spearman of proxy
   vs exact on a tiny synthetic codebook; faithful and unfaithful cases via mocked
   calculators).
4. [ ] application: `EvaluationSuiteUseCase` + tests (runs gate then cells; raises
   on unfaithful scaled cell; mocks the evaluate/fidelity use cases).
5. [ ] interface: `eval --distance/--max-samples`, `eval_suite` CLI, configs, tox
   env, architecture-test wiring, README table.
6. [ ] integration (marked, not unit): run AG News full/large + IMDB + 20NG with
   the sliced proxy; confirm the fidelity gate passes; commit
   `reports/eval_results.md` and the README numbers.
7. [ ] run `tox`; confirm 8 gates + 100% coverage; reproduce one committed number
   end-to-end from its config.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names, no
network in unit tests (mock the embedding adapter and dataset repository; use a
small synthetic `ColorCodebook.create_uniform_grid`). Key tests:

- **Sliced distance (real `ot`):** identical histograms → 0; distinct histograms →
  positive and symmetric; two calls with the same seed → identical value; rejects
  a histogram from a different codebook size.
- **Fidelity (pure-ish):** with mocked proxy/exact calculators returning known
  distance vectors, Spearman is computed correctly and `is_faithful` flips at the
  threshold; an anti-correlated proxy is reported unfaithful.
- **Suite:** a faithful proxy lets all cells run; an unfaithful scaled cell raises
  rather than reporting a number (mocked use cases — no datasets downloaded).
- **CLI branches:** `--distance` selects the right calculator class; `eval_suite`
  writes the report and prints the table (mock the use cases, real `tmp_path`).
- **Scaled numbers:** produced by the integration run, committed to
  `reports/eval_results.md` — not asserted in unit tests (no network / model
  downloads in CI unit runs).

## Observability Plan

`correlation-id` logging: fidelity use case logs `{proxy, spearman,
accuracy_delta, pair_count, is_faithful}`; suite logs per-cell `{dataset, method,
distance, budget, accuracy, macro_f1, seconds}`. No new metrics/tracing.

## Risks and Mitigations

- **Proxy silently changes the result.** Mitigation: the fidelity gate is
  mandatory and fails loudly below threshold; the committed table records which
  distance produced each number.
- **Full test set too slow for CI.** Mitigation: run the largest budget that fits
  and record it; the unit suite never runs the full dataset (integration-marked).
- **Sliced-Wasserstein variance from random projections.** Mitigation: fixed seed
  + enough projections (≥100); determinism asserted by a repeat-call test.
- **Re-embedding dominates wall-clock.** Mitigation: encode each document once per
  run; note a persistent cache as a future option (Open Questions).
- **Layer leakage.** Mitigation: `ot` stays in infrastructure behind
  `DistanceCalculator`; an architecture test forbids `ot`/`scipy` in the domain.
- **Determinism drift across `ot`/`hnswlib` versions.** Mitigation: pinned seeds
  and `set_num_threads(1)` (already set in the classifier); record library
  versions alongside the committed numbers.
