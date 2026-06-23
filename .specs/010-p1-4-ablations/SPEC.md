# Feature: Ablations the blog names

## Overview

The blog post poses, but never runs, the two ablations that decide whether the
color-of-meaning pipeline's design choices matter: how finely Lab space is
quantized, and which distance metric compares the resulting color histograms.
This step adds a single reproducible sweep that crosses **quantization level**
(1024-bin grid vs 4096-bin grid vs a learned VQ codebook from
`007-p1-1-learned-vq-codebook`) with **distance metric** (Lab Earth-Mover from
`001-p0-1-lab-emd-distance` vs Jensen-Shannon vs a new cosine-on-histograms
calculator), and reports two numbers per cell: downstream classification
**Accuracy / Macro-F1** and the **structure-preservation Spearman correlation**
from `005-p0-5-determinism-structure-metric`.

This is an orchestration step, not new science. It deliberately reuses the
existing evaluation path end-to-end: `EvaluateUseCase`
(`src/colors_of_meaning/application/use_case/evaluate_use_case.py`) drives a
`ColorHistogramClassifier`
(`src/colors_of_meaning/infrastructure/evaluation/color_histogram_classifier.py`)
that is parameterised by a `DistanceCalculator`
(`src/colors_of_meaning/domain/service/distance_calculator.py`) and a
`ColorCodebook` (`src/colors_of_meaning/domain/model/color_codebook.py`). The
only genuinely new production code is one `DistanceCalculator` implementation
(cosine over histograms) and a thin application use case plus CLI command that
re-run the existing single-point evaluation once per `(codebook, metric)`
combination and collect the results into one table.

Today the single-point `eval` command
(`src/colors_of_meaning/interface/cli/eval.py`) hardcodes the distance
calculator (`WassersteinDistanceCalculator()` at line 87) and takes exactly one
`codebook_path` (default `codebook_4096`) and one `method`; there is no way to
vary the metric, no way to vary quantization, and no structure-preservation
number in its output (`EvaluationResult` carries accuracy, macro_f1,
recall_at_k, mrr, bits_per_token only). This step closes that gap without
disturbing the existing single-point command.

The numbers this sweep produces are only meaningful once the upstream chain has
landed, exactly as for `006-p0-6-end-to-end-agnews-table`: a real Lab EMD
(`001`), a real training objective (`002-p0-2-structure-preserving-training`),
seeded sampling (`003-p0-3-shuffle-stratify-sampling`), an `eval` that loads
every mapper variant with `ef >= num_candidates`
(`004-p0-4-eval-mapper-coverage-ef-fix`), honoured seeding plus the
structure-preservation evaluator (`005`), and the learned VQ codebook (`007`).
The sweep therefore **depends on all of P0 and on P1-1** and must be run after
they merge.

## User Stories

- As an ML researcher, I want one command that sweeps quantization level against
  distance metric so that I can report each design choice's effect on accuracy
  instead of asserting it from the blog.
- As an ML researcher, I want each sweep cell to report the
  structure-preservation correlation alongside accuracy so that I can see
  whether finer quantization or a different metric actually preserves semantic
  geometry, not just classification accuracy.
- As a researcher, I want the sweep to reuse the existing single-point
  evaluation path so that a sweep cell and a standalone `eval` run on the same
  `(codebook, metric)` produce the same accuracy.
- As a maintainer, I want cosine-on-histograms to implement the existing
  `DistanceCalculator` port so that it slots into `ColorHistogramClassifier`
  with no special-casing and no new dependency.
- As a reviewer, I want the sweep to emit a machine-readable artifact (and a
  printed table) so that the published ablation grid is attributable to a
  specific, version-controlled run.

## Acceptance Criteria

- [ ] Given a set of codebooks (1024-bin, 4096-bin, learned) and a set of
  metrics (EMD, Jensen-Shannon, cosine), when the sweep use case runs, then it
  produces one result row per `(codebook, metric)` combination.
- [ ] Given a single `(codebook, metric)` combination, when the sweep evaluates
  it, then it reuses `EvaluateUseCase` so the reported Accuracy / Macro-F1 match
  a standalone `eval` run with the same codebook and metric within numerical
  tolerance.
- [ ] Given a `(codebook, metric)` combination, when the sweep evaluates it,
  then the row records the structure-preservation Spearman correlation from the
  `005` evaluator alongside Accuracy and Macro-F1.
- [ ] Given two color histograms over the same codebook, when the cosine
  calculator computes their distance, then it returns `0.0` for identical
  normalized histograms and a strictly larger value for differing ones, in the
  closed interval `[0, 1]`.
- [ ] Given two histograms with differing bin counts, when the cosine
  calculator computes their distance, then it raises `ValueError` (matching
  `WassersteinDistanceCalculator` and `JensenShannonDistanceCalculator`).
- [ ] Given a requested metric name on the CLI, when it is `wasserstein`,
  `jensen_shannon`, or `cosine`, then the matching `DistanceCalculator`
  implementation is constructed and injected; when it is unknown, then a
  `ValueError` is raised.
- [ ] Given a completed sweep, when the command finishes, then it writes a
  machine-readable artifact of the full grid and prints a human-readable table
  to stdout.
- [ ] Given the configured seed and committed config, when the sweep is re-run
  on the same environment, then the grid is reproduced within a stated
  tolerance (depends on `003`/`005` determinism).
- [ ] Given this step, when `tox` is run, then all eight gates pass and coverage
  remains 100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No new ports. The sweep reuses existing abstractions only:

- `domain/service/distance_calculator.py` — the cosine calculator implements
  this existing `DistanceCalculator` ABC (`compute_distance`, `metric_name`); no
  change to the port.
- `domain/service/classifier.py`, `domain/service/metrics_calculator.py`,
  `domain/service/structure_preservation_evaluator.py` (introduced by `005`),
  `domain/repository/dataset_repository.py`,
  `domain/repository/color_codebook_repository.py` — reused unchanged.
- `domain/model/color_codebook.py` — quantization level is `ColorCodebook.num_bins`;
  the sweep varies *which* codebook is loaded, not the model. Note: the grid
  factory `create_uniform_grid` produces `bins_per_dimension ** 3` bins, so a
  "1024-bin" grid is not a perfect cube (10^3 = 1000, not 1024); how the
  1024-level grid is obtained is an Open Question. No change to the model.
- A small domain result type for one sweep cell (e.g. an `AblationResult`
  dataclass under `domain/model/`) MAY be added to carry
  `(codebook_label, metric_name, EvaluationResult, structure_correlation)` as a
  typed value object rather than a dict; whether to add it or reuse
  `EvaluationResult` plus a tuple is an Open Question. If added it is a frozen
  dataclass with no framework imports, mirroring `EvaluationResult`
  (`domain/model/evaluation_result.py`).

### Application Layer (`src/colors_of_meaning/application/`)

- **New** `application/use_case/ablation_sweep_use_case.py`: an
  `AblationSweepUseCase` that, given the iterables of codebooks and distance
  calculators (plus the dataset repository, metrics calculator, embedding
  adapter and structure-preservation evaluator it needs), constructs a
  `ColorHistogramClassifier` per `(codebook, metric)`, runs the existing
  `EvaluateUseCase` for each, also computes the structure-preservation
  correlation for that codebook, and returns the collected grid of results. It
  orchestrates only; it imports domain abstractions
  (`Classifier`, `MetricsCalculator`, `DistanceCalculator`,
  `DatasetRepository`, `StructurePreservationEvaluator`) and the existing
  `EvaluateUseCase`, never torch/scipy/sklearn or any infrastructure concrete.
- `application/use_case/evaluate_use_case.py` — reused unchanged; the sweep
  delegates a full `execute(...)` call per cell so accuracy parity is automatic.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- **New** `infrastructure/ml/cosine_histogram_distance_calculator.py`: a
  `CosineHistogramDistanceCalculator(DistanceCalculator)` returning a cosine
  distance (`1 - cosine_similarity`) over the two normalized histograms,
  validating equal `num_bins` first, mirroring the structure and error handling
  of `infrastructure/ml/wasserstein_distance_calculator.py` and
  `infrastructure/ml/jensen_shannon_distance_calculator.py`. It lives beside
  them in `infrastructure/ml/` and uses `numpy` (already a dependency) or
  `scipy.spatial.distance.cosine` (scipy already a dependency, same import
  pattern as the Jensen-Shannon calculator); `metric_name()` returns `"cosine"`.
- `infrastructure/evaluation/color_histogram_classifier.py`,
  `infrastructure/evaluation/sklearn_metrics_calculator.py`,
  `infrastructure/embedding/sentence_embedding_adapter.py`,
  `infrastructure/persistence/file_color_codebook_repository.py`,
  `infrastructure/evaluation/structure_preservation_evaluator.py` (from `005`),
  and the learned-codebook factory from `007` — all reused unchanged.
- No new third-party dependency: cosine is covered by numpy/scipy already in
  `setup.cfg install_requires`, so `pip-audit` (a tox gate) stays green.

### Interface Layer (`src/colors_of_meaning/interface/`)

- **Preferred:** a new CLI entry point `interface/cli/ablate.py` (an `AblateArgs`
  tyro dataclass) that selects the dataset, the list of codebook paths/labels
  (1024-bin, 4096-bin, learned), the list of metrics, the model path and
  mapper-type (reusing the mapper factory introduced by `004`), builds the
  matching `DistanceCalculator` and `ColorCodebook` per combination via factory
  functions, constructs `AblationSweepUseCase`, runs it, prints the grid and
  writes the artifact. This keeps the existing single-point `eval.py` command
  untouched. Whether to add a new command or extend `eval.py` with list-valued
  `--codebooks` / `--metrics` flags is an Open Question.
- `interface/cli/eval.py` — unchanged under the preferred design; its
  single-point selection (`method`, `codebook_path`, `model_path`) is the
  template the sweep's per-cell wiring follows, and the metric factory used by
  the sweep MAY be back-ported into `eval.py` so it too can choose
  Jensen-Shannon or cosine (Open Question).

### Shared Layer

No required change. The sweep reuses the existing `synesthetic_config` schema
loaded by `SynestheticConfig.from_yaml`
(`src/colors_of_meaning/shared/synesthetic_config.py`); `DistanceConfig.metric`
already exists and the sweep overrides it per cell. If the artifact path or the
default codebook/metric lists are configured rather than passed as CLI flags, a
small optional `AblationConfig` block could be added to the schema (Open
Question). `shared/lab_utils.py` is reused only indirectly via the codebooks.

## API Contracts

No changes. This step touches the CLI/evaluation path only; no FastAPI
controllers or DTOs are added or modified. (HTTP surface: No changes.)

## CLI Impact

Preferred: one new command, `tox -e ablate -- ...` (a new tox environment
forwarding posargs to `interface/cli/ablate.py`, matching the existing
`train` / `encode` / `eval` environments). `AblateArgs` fields (proposed):

- `--config` (default `configs/base.yaml`)
- `--dataset` (`ag_news` | `imdb` | `newsgroups`)
- `--codebooks` (list of `label=path` pairs, e.g.
  `grid1024=codebook_1024 grid4096=codebook_4096 learned=codebook_learned`)
- `--metrics` (list from `wasserstein`, `jensen_shannon`, `cosine`)
- `--model-path`, `--mapper-type` (reuse the `004` factory)
- `--k-neighbors`
- `--output-path` (artifact destination)

The command prints a table with columns
`codebook | metric | accuracy | macro_f1 | structure_correlation` and writes the
same grid to `--output-path`. The existing `eval.py`, `encode.py`, `train.py`,
`compare.py`, `compress.py`, `query.py`, `visualize.py` commands are unchanged.

## Dependency Injection

- CLI factory style (matching `_create_classifier` / `_create_color_classifier`
  in `eval.py` and `_create_color_mapper` in `train.py`): factory functions
  build (a) the `DistanceCalculator` for a metric name —
  `wasserstein -> WassersteinDistanceCalculator`,
  `jensen_shannon -> JensenShannonDistanceCalculator`,
  `cosine -> CosineHistogramDistanceCalculator`, else `ValueError`; (b) the
  `ColorCodebook` for a codebook label via `FileColorCodebookRepository().load`;
  (c) the mapper via the `004` mapper factory. These are injected into
  `AblationSweepUseCase` together with `EvaluateUseCase`,
  `SklearnMetricsCalculator`, the dataset adapter, the
  `SentenceEmbeddingAdapter`, and the `SpearmanStructurePreservationEvaluator`
  from `005`.
- `AblationSweepUseCase` depends only on domain abstractions and the existing
  `EvaluateUseCase`; it never instantiates infrastructure concretes itself.
- API Lagom `Container()` in `interface/api/main.py`: no binding required; the
  sweep is CLI-only.
- No new dependency is introduced; `pip-audit` (tox) stays green.

## Observability

- Structured log (with `correlation-id`) of the resolved sweep matrix (codebook
  labels x metric names), the seed, and the dataset at startup.
- Log each cell as it completes: codebook label, metric name, accuracy,
  macro_f1, and structure-preservation correlation.
- Emit accuracy and structure-preservation correlation per cell as metrics
  (gauges tagged by codebook and metric) so cells are comparable across runs.
- Reuse the existing per-method progress prints from `eval.py`'s
  `_print_results` style for the final table.

## Open Questions

1. **Report artifact format:** CSV, JSON, or a Markdown table committed next to
   the README ablation section? A machine-readable form (CSV/JSON) plus a printed
   table is the working assumption; the committed location and whether it is
   version-controlled or a release asset mirror the same question in `006`.
2. **New CLI command vs extending `eval`:** add `interface/cli/ablate.py` (+ a
   new tox env) — preferred, keeps the single-point `eval` contract stable — or
   extend `EvalArgs` with list-valued `--codebooks` / `--metrics` and branch
   inside `eval.py`? Extending `eval` risks bloating a command whose single-point
   behaviour `006` documents.
3. **How is the 1024-level quantization obtained?** `create_uniform_grid`
   yields `bins_per_dimension ** 3` bins, and 1024 is not a perfect cube; options
   are a `bins_per_dimension=10` grid relabelled "~1024" (1000 bins), an exact
   1024-center learned codebook from `007`, or treating "1024 vs 4096" purely as
   two learned-codebook sizes. This must be pinned before the sweep is run.
4. **Cosine semantics on sparse histograms:** color histograms are largely zero;
   should cosine operate on the raw normalized histogram, or on a smoothed
   histogram like Jensen-Shannon's `smoothing_epsilon` path? Define whether
   cosine needs the same `1e-8` floor for numerical stability.
5. **Does the sweep re-encode per metric, or encode once per codebook?**
   Encoding depends only on the codebook (and mapper), not the metric, so
   histograms could be computed once per codebook and reused across the three
   metrics; the simplest design reuses `EvaluateUseCase` wholesale per cell
   (re-encoding each time) at the cost of redundant embedding work. Trading
   parity-by-construction against runtime is an Open Question.
6. **Structure-preservation pairs source:** the `005` evaluator scores embeddings
   against Lab colors for a given mapper/codebook; confirm the sweep feeds it the
   same held-out split per cell so correlations are comparable across codebooks,
   coordinating the seeded split with `003`.
7. **Which mapper variant(s) does the sweep cover?** A single mapper across all
   cells isolates the quantization/metric effect; sweeping mapper variants too
   would multiply the grid. Recommend fixing the mapper (the canonical one chosen
   in `006`) and noting variant sweeps as future work.
