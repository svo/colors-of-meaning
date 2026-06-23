# Plan: ablations-the-blog-names

## Implementation Strategy

Add the smallest amount of production code that lets one command sweep
**quantization level** (1024-bin grid vs 4096-bin grid vs learned VQ codebook
from `007-p1-1-learned-vq-codebook`) against **distance metric** (Lab EMD from
`001-p0-1-lab-emd-distance` vs Jensen-Shannon vs cosine) and report Accuracy,
Macro-F1, and the structure-preservation Spearman correlation from
`005-p0-5-determinism-structure-metric` per cell.

The design reuses the existing single-point evaluation path wholesale: per
`(codebook, metric)` cell the sweep builds a `ColorHistogramClassifier`
(`infrastructure/evaluation/color_histogram_classifier.py`) parameterised by the
chosen `ColorCodebook` and `DistanceCalculator`, then runs the existing
`EvaluateUseCase` (`application/use_case/evaluate_use_case.py`). Accuracy parity
with standalone `eval` is therefore by construction. The only new production code
is: one `DistanceCalculator` implementation (cosine over histograms), one
application sweep use case, factory functions for metric/codebook selection, and
a thin CLI command. No new domain port, no new third-party dependency (cosine is
covered by numpy/scipy already in `setup.cfg`).

Build TDD, domain/infra-first: the cosine calculator and its contract test
land first (it has the only non-trivial new numerics), then the application sweep
use case against mocks, then the CLI wiring. This step is sequenced **after all
of P0 and after P1-1**; until `001`, `002`, `003`, `004`, `005`, and `007` merge,
the swept numbers are not interpretable (same rationale as `006`).

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- No new port. The cosine calculator implements the existing
  `domain/service/distance_calculator.py` ABC (`compute_distance`,
  `metric_name`); the port is unchanged.
- Optional new value object `domain/model/ablation_result.py`: a frozen
  dataclass `AblationResult(codebook_label: str, metric_name: str, result:
  EvaluationResult, structure_correlation: float)` with no framework imports,
  mirroring `domain/model/evaluation_result.py`. Whether to add it or return a
  list of `(label, metric, EvaluationResult, float)` tuples is Open Question 8;
  the plan assumes the dataclass for type clarity and testability.
- `domain/model/color_codebook.py`, `domain/service/classifier.py`,
  `domain/service/metrics_calculator.py`,
  `domain/service/structure_preservation_evaluator.py` (from `005`),
  `domain/repository/dataset_repository.py` — inputs only, No changes.

### Application Layer (`src/colors_of_meaning/application/`)

- New file `application/use_case/ablation_sweep_use_case.py`:
  `AblationSweepUseCase`. Constructor takes the dataset repository, metrics
  calculator, embedding adapter, structure-preservation evaluator, an
  `EvaluateUseCase` factory (or the pieces to build one per cell), and the
  iterables of `(codebook_label, ColorCodebook)` and
  `(metric_name, DistanceCalculator)`. `execute(...)` loops the cross product,
  for each cell builds a `ColorHistogramClassifier`, runs `EvaluateUseCase`,
  computes the structure-preservation correlation for that codebook, and appends
  an `AblationResult`. Imports domain abstractions and the existing
  `EvaluateUseCase` only — no torch/scipy/sklearn, no infrastructure concrete.
- `application/use_case/evaluate_use_case.py` — reused unchanged.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- New file `infrastructure/ml/cosine_histogram_distance_calculator.py`:
  `CosineHistogramDistanceCalculator(DistanceCalculator)`. Validates
  `doc1.num_bins == doc2.num_bins` (raise `ValueError` otherwise, matching
  `wasserstein_distance_calculator.py` / `jensen_shannon_distance_calculator.py`),
  returns `1 - cosine_similarity` over the two normalized histograms in `[0, 1]`,
  `metric_name()` returns `"cosine"`. Uses numpy or
  `scipy.spatial.distance.cosine` (both already declared). Lives in
  `infrastructure/ml/` beside the other two calculators.
- `infrastructure/evaluation/color_histogram_classifier.py`,
  `infrastructure/evaluation/sklearn_metrics_calculator.py`,
  `infrastructure/embedding/sentence_embedding_adapter.py`,
  `infrastructure/persistence/file_color_codebook_repository.py`,
  `infrastructure/evaluation/structure_preservation_evaluator.py` (from `005`),
  the learned-codebook factory from `007` — reused unchanged.

### Interface Layer (`src/colors_of_meaning/interface/`)

- New file `interface/cli/ablate.py`: `AblateArgs` (tyro dataclass) with
  `--config`, `--dataset`, `--codebooks` (list of `label=path`), `--metrics`
  (list), `--model-path`, `--mapper-type`, `--k-neighbors`, `--output-path`.
  Factory functions `_create_distance_calculator(metric_name)` and
  `_load_codebook(label_path)`; reuse the `004-p0-4` mapper factory for
  `--mapper-type`. `main` builds the codebook list and metric list, constructs
  `AblationSweepUseCase`, runs it, prints the table, writes the artifact.
- `interface/cli/eval.py` — unchanged under the preferred design (Open Question
  9). The metric factory MAY later be shared with `eval.py`.

### Shared Layer (`src/colors_of_meaning/shared/`)

- `shared/synesthetic_config.py` — No changes required; `DistanceConfig.metric`
  already exists and is overridden per cell, and `CodebookConfig` carries
  `bins_per_dimension` / `num_bins`. An optional `AblationConfig` block (default
  codebook labels, metric list, artifact path) is deferred to Open Question 1.
- `shared/lab_utils.py` — reused only indirectly; No changes.

## Dependency Injection

- CLI factory style (matching `_create_classifier` / `_create_color_classifier`
  in `eval.py`):
  - `_create_distance_calculator`: `wasserstein -> WassersteinDistanceCalculator`,
    `jensen_shannon -> JensenShannonDistanceCalculator`,
    `cosine -> CosineHistogramDistanceCalculator`, unknown -> `ValueError`.
  - `_load_codebook`: `FileColorCodebookRepository().load(path)`,
    `FileNotFoundError` when absent (matching `eval.py` line 82-83).
  - mapper via the `004` factory.
- These concretes are injected into `AblationSweepUseCase`, which depends only on
  the domain abstractions (`DistanceCalculator`, `Classifier`,
  `MetricsCalculator`, `DatasetRepository`, `StructurePreservationEvaluator`) and
  the existing `EvaluateUseCase`.
- API Lagom `Container()` (`interface/api/main.py`): no binding required.
- No new third-party dependency; scipy/numpy already declared. `pip-audit` (tox)
  stays green.

## Task List

1. [ ] domain: (optional) add frozen `AblationResult` dataclass in
   `domain/model/ablation_result.py` carrying codebook label, metric name,
   `EvaluationResult`, and structure correlation (no framework imports).
2. [ ] infrastructure: implement `CosineHistogramDistanceCalculator` in
   `infrastructure/ml/cosine_histogram_distance_calculator.py`, validating equal
   `num_bins`, returning `1 - cosine_similarity` in `[0, 1]`, `metric_name() ==
   "cosine"`.
3. [ ] application: implement `AblationSweepUseCase` in
   `application/use_case/ablation_sweep_use_case.py` iterating the
   `(codebook, metric)` cross product, delegating each cell to `EvaluateUseCase`
   and computing the `005` structure correlation per codebook, returning the
   collected `AblationResult` grid; domain-only imports.
4. [ ] interface: add `_create_distance_calculator` and `_load_codebook`
   factories plus `AblateArgs` and `main` in `interface/cli/ablate.py`, reusing
   the `004` mapper factory; print the table and write the artifact.
5. [ ] interface: add a `tox -e ablate` environment forwarding posargs to
   `interface/cli/ablate.py` (mirror the `eval` env in `tox.ini`).
6. [ ] tests: cosine returns `0.0` for identical normalized histograms.
7. [ ] tests: cosine returns a value in `[0, 1]` strictly greater than `0` for
   differing histograms.
8. [ ] tests: cosine raises `ValueError` when bin counts differ.
9. [ ] tests: `CosineHistogramDistanceCalculator.metric_name` returns
   `"cosine"`.
10. [ ] tests: `_create_distance_calculator` returns the matching calculator for
    each of `wasserstein`, `jensen_shannon`, `cosine` (one test per name).
11. [ ] tests: `_create_distance_calculator` raises `ValueError` for an unknown
    metric.
12. [ ] tests: `AblationSweepUseCase.execute` produces one result per
    `(codebook, metric)` combination (mock `EvaluateUseCase` + stub evaluator).
13. [ ] tests: a sweep cell's `EvaluationResult` equals the standalone
    `EvaluateUseCase` result for the same codebook and metric (parity, with a
    stub classifier/metrics).
14. [ ] tests: each `AblationResult` carries the structure-preservation
    correlation returned by the stubbed `005` evaluator.
15. [ ] tests: `_load_codebook` raises `FileNotFoundError` when the codebook is
    absent (patched repository).
16. [ ] tests: the CLI `main` writes an artifact and prints a row per cell
    (patched use case, `tmp_path` for the artifact).
17. [ ] tests: pytest-archon rule confirms cosine's scipy/numpy import stays in
    `infrastructure.ml` and `AblationSweepUseCase` imports no infrastructure;
    extend `tests/colors_of_meaning/test_synesthetic_architecture.py` and add
    `ablate` to the CLI-uses-use-case rule.
18. [ ] tests: add `__init__.py` to every new test package mirroring the source
    tree.
19. [ ] verify: run `tox` (8 gates) for 100% coverage, flake8, black, bandit,
    semgrep, mypy, xenon, radon, pip-audit.

## Testing Strategy

- One logical assertion per test; ML/numerical tests may group related asserts on
  the same result (e.g. value plus `[0, 1]` bound). Names follow
  `test_should_<behaviour>_when_<condition>`.
- `assertpy` (`assert_that`) for the entity-style `AblationResult` tests and CLI
  artifact-shape tests; plain `assert` / `pytest.raises` for the cosine numerics
  and the `ValueError` / `FileNotFoundError` paths.
- Cosine tests build `ColoredDocument` instances with hand-chosen normalized
  histograms (identical, orthogonal, differing) and assert the distance value and
  its bound; no network calls, no embeddings.
- Sweep tests inject a mock `EvaluateUseCase` returning scripted
  `EvaluationResult`s and a stub `StructurePreservationEvaluator` returning a
  scripted correlation, asserting cell count, parity, and that the correlation is
  carried through — no real datasets or models.
- CLI tests patch the use case and dataset/codebook factories, run `main` over a
  `tmp_path` artifact, and assert the artifact is written and a row is printed
  per cell; no real `SentenceEmbeddingAdapter` (no model download in unit tests).
- Architecture: extend `tests/colors_of_meaning/test_synesthetic_architecture.py`
  (pytest-archon) so `domain.*` imports no other layer, `application.*`
  (including `AblationSweepUseCase`) imports no infrastructure, and
  `interface.cli.ablate` imports `application.use_case.*` (add to the existing
  CLI-uses-use-case `archrule.match(...)` list).
- New tests mirror source under
  `tests/colors_of_meaning/infrastructure/ml/`,
  `tests/colors_of_meaning/application/use_case/`,
  `tests/colors_of_meaning/interface/cli/`, and (if added)
  `tests/colors_of_meaning/domain/model/`, each with `__init__.py`.
- No comments in source or tests. Final verification via `tox` only (never bare
  pytest); 100% coverage required.

## Observability Plan

- Structured log (with `correlation-id`) of the resolved sweep matrix (codebook
  labels x metric names), the seed, and the dataset at startup.
- Log each cell on completion: codebook label, metric name, accuracy, macro_f1,
  structure-preservation correlation.
- Emit per-cell accuracy and structure correlation as gauge metrics tagged by
  codebook label and metric name.
- Reuse the existing `_print_results`-style stdout summary from `eval.py` for the
  final grid table.

## Risks and Mitigations

- **Accuracy parity drift between a sweep cell and standalone `eval`.** Mitigation:
  delegate each cell to the *same* `EvaluateUseCase` rather than reimplementing
  the loop; parity test (task 13) pins it.
- **Redundant embedding work.** Reusing `EvaluateUseCase` per cell re-encodes
  documents for every metric even though encoding depends only on the codebook.
  Mitigation: accept the simplest parity-by-construction design first; if runtime
  is a problem, encode once per codebook and reuse across metrics (Open Question
  5) — but only behind a test proving identical results.
- **Ambiguous 1024-level quantization.** `create_uniform_grid` yields
  `bins_per_dimension ** 3` and 1024 is not a perfect cube. Mitigation: pin the
  1024-level definition (relabelled 10^3=1000 grid, or an exact 1024-center
  learned codebook from `007`) before running (Open Question 3); the sweep code
  is agnostic since it loads whatever codebook the label points at.
- **Cosine instability on sparse histograms.** Mostly-zero histograms can make
  cosine numerically fragile. Mitigation: decide on a `1e-8` floor mirroring the
  Jensen-Shannon `smoothing_epsilon` (Open Question 4); cover the all-zero-overlap
  case with a test.
- **Upstream sequencing.** The grid is meaningless until `001`, `002`, `003`,
  `004`, `005`, and `007` land. Mitigation: treat this as a post-P0/P1-1 delivery
  step (as `006` does for P0); the code and tests use synthetic inputs so they are
  developed and verified independently of a real trained run.
- **Layer leakage.** Building classifiers/calculators inside the use case could
  pull infrastructure into application. Mitigation: pass codebooks and
  `DistanceCalculator`s in already-constructed via CLI factories; keep
  `AblationSweepUseCase` import-clean and assert it with pytest-archon (task 17).
