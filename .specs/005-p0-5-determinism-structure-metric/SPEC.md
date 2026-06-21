# Feature: Determinism and Structure-Preservation Metric

## Overview

Training in `Colors of Meaning` is non-deterministic and offers no direct measurement of the blog's central claim that "similar meanings map to similar colors". Two problems are addressed together:

1. **Determinism.** `TrainingConfig.seed` (default `42`, in `src/colors_of_meaning/shared/synesthetic_config.py`) is never applied anywhere in `src/`. The color mappers call `torch.randperm` for batch shuffling, `torch.rand` for target generation, and rely on default-seeded `nn.Linear` weight initialisation. Consequently identical seeds do not reproduce identical Lab outputs, so runs are not comparable and checkpoint selection is noise-driven.

2. **Structure-preservation metric.** Quality is currently inferred indirectly from downstream classification accuracy. There is no metric that directly quantifies whether embedding-space similarity is preserved in color space. We introduce a metric reporting the Spearman rank correlation between embedding-space cosine similarity and color-space distance over sampled pairs. Because closer embeddings should yield closer colors, a well-aligned mapper produces a strongly negative correlation; the sign-normalised magnitude (or `-correlation`) is the score used to drive best-checkpoint selection.

This step pairs with `002-p0-2-structure-preserving-training`: P0-2 changes the training objective so that meaning-similar inputs map to color-similar outputs, and this metric (P0-5) is the held-out instrument that measures whether that objective is being achieved. Determinism is a prerequisite for both: without a honoured seed, the metric's checkpoint comparison is meaningless.

The new evaluator lives in `src/colors_of_meaning/infrastructure/evaluation/structure_preservation_evaluator.py` behind a new `domain/service` port (`StructurePreservationEvaluator` ABC). Spearman / scipy stays in infrastructure to keep the domain pure (mirrors `WassersteinDistanceCalculator` in `infrastructure/ml/`, which imports `scipy.stats`).

## User Stories

- As an ML researcher, I want identical seeds to reproduce identical Lab outputs so that training runs are comparable and results are reproducible.
- As an ML researcher, I want a metric that directly measures whether similar meanings map to similar colors so that I can report the blog's claim instead of inferring it from downstream accuracy.
- As an ML researcher, I want best-checkpoint selection driven by the structure-preservation metric so that the saved model is the one that best preserves semantic structure, not merely the last epoch.
- As a maintainer, I want the structure-preservation metric defined behind a domain port with the scipy implementation in infrastructure so that the domain layer stays free of numerical-library dependencies.
- As a maintainer, I want an optional strict-determinism flag so that I can trade speed for bit-exact reproducibility when auditing results.

## Acceptance Criteria

- [ ] Given two training runs configured with the same `TrainingConfig.seed`, when each trains the same mapper on the same embeddings, then both produce identical Lab outputs for the same input embedding.
- [ ] Given a configured `seed`, when training bootstrap runs, then `torch.manual_seed(seed)` and `numpy.random.seed(seed)` are applied and a seeded `torch.Generator` is passed to every `torch.randperm` / `torch.rand` call inside the mappers.
- [ ] Given the strict-determinism flag is enabled, when training bootstrap runs, then `torch.use_deterministic_algorithms(True)` is set; when the flag is disabled (default), then it is not set.
- [ ] Given embeddings and their corresponding Lab colors, when the `StructurePreservationEvaluator` computes its score, then it returns a Spearman correlation in the closed interval `[-1, 1]`.
- [ ] Given embeddings whose pairwise cosine similarities are perfectly inversely ranked against their color-space distances, when the evaluator runs, then it returns a correlation of `-1.0` within numerical tolerance.
- [ ] Given embeddings whose pairwise cosine similarities have no rank relationship to color-space distances, when the evaluator runs, then the magnitude of the returned correlation is near `0`.
- [ ] Given a multi-epoch training run, when checkpoint selection runs, then the persisted "best" weights are those of the epoch maximising the structure-preservation score (sign-normalised so that better structure preservation is a larger value).
- [ ] Given fewer than two evaluation pairs, when the evaluator runs, then it raises `ValueError` rather than returning an undefined correlation.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

A new port `StructurePreservationEvaluator` (`domain/service/structure_preservation_evaluator.py`), an `ABC` mirroring the style of `domain/service/distance_calculator.py` and `domain/service/metrics_calculator.py` (abstract methods raising `NotImplementedError`, no framework imports). Proposed abstract methods:

- `evaluate(self, embeddings: npt.NDArray, lab_colors: List[LabColor]) -> float` — returns the Spearman correlation in `[-1, 1]` between embedding-space cosine similarity and color-space distance over sampled pairs.
- `metric_name(self) -> str` — returns a stable identifier (e.g. `"structure_preservation_spearman"`), mirroring `DistanceCalculator.metric_name`.

Inputs are existing domain types: `List[LabColor]` (`domain/model/lab_color.py`, exposing `to_tuple()` / `l,a,b`) and the raw embedding array. The port MUST NOT import torch, scipy, or sklearn. Color-space distance over `LabColor` is Euclidean in Lab (Delta E 76); whether to reuse a shared Lab utility is an Open Question.

### Application Layer (`src/colors_of_meaning/application/`)

`TrainColorMappingUseCase` (`application/use_case/train_color_mapping_use_case.py`) gains the responsibility of best-checkpoint selection. Today `execute` calls `color_mapper.train(...)` then unconditionally `color_mapper.save_weights(model_name)`. It will be extended to:

- Receive a `StructurePreservationEvaluator` (the new port) via constructor injection, alongside the existing `ColorMapper` and `ColorCodebookRepository`.
- Receive held-out evaluation embeddings (and the `seed`) so it can score checkpoints.
- Drive epoch-wise (or post-train candidate) evaluation and persist the highest-scoring weights.

The application layer imports domain only; it depends on the `ColorMapper`, `StructurePreservationEvaluator`, and `ColorCodebookRepository` abstractions, never on torch or scipy. How per-epoch checkpoint scoring is exposed without leaking torch into application is an Open Question (candidate: extend the `ColorMapper` port with a callback / per-epoch hook, or score a fixed set of post-training candidate checkpoints).

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- **New** `infrastructure/evaluation/structure_preservation_evaluator.py`: a `SpearmanStructurePreservationEvaluator(StructurePreservationEvaluator)` implementing the port using `scipy.stats.spearmanr` (scipy already in `setup.cfg install_requires`; same pattern as `infrastructure/ml/wasserstein_distance_calculator.py`). It builds the pairwise embedding cosine-similarity vector and the matching color-space distance vector over upper-triangle pairs (optionally subsampled with a seeded generator for large batches) and returns `spearmanr(...).correlation` cast to `float`. Lives beside the existing classifiers in `infrastructure/evaluation/`.
- **Seeding** in the three mappers (`infrastructure/ml/pytorch_color_mapper.py`, `structured_pytorch_color_mapper.py`, `supervised_pytorch_color_mapper.py`): honour the seed by calling `torch.manual_seed(seed)` / `numpy.random.seed(seed)` before weight init, seeding a `torch.Generator` (on `self.device`) and passing it to every `torch.randperm(..., generator=g)` and `torch.rand(..., generator=g)` call (`pytorch_color_mapper.py` lines 94, 128-130; `structured` line 139; `supervised` lines 132, 204-206). The optional `torch.use_deterministic_algorithms(True)` is applied behind a flag. How the seed reaches the mappers (constructor argument vs. a shared seeding helper invoked at bootstrap) is detailed in PLAN; a thin `shared/` seeding helper is preferred so all randomness sources are seeded in one place.

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/cli/train.py` (the bootstrap, `main` / `_execute_training`) applies the configured seed once at startup before any model construction or embedding, and exposes a new strict-determinism flag on `TrainArgs` (e.g. `deterministic: bool = False`) threaded into the seeding helper. It also constructs and injects the `SpearmanStructurePreservationEvaluator` into `TrainColorMappingUseCase`, and supplies held-out evaluation embeddings for checkpoint selection (a held-out slice of the encoded embeddings, or a re-encoded eval split — see Open Questions). The CLI may print the final structure-preservation correlation so P0-6 can record it.

No API controller changes are required for this step.

### Shared Layer

A new seeding helper in `src/colors_of_meaning/shared/` (e.g. `determinism.py` exposing `seed_everything(seed: int, deterministic: bool = False)`) that wraps `torch.manual_seed`, `numpy.random.seed`, optional `torch.use_deterministic_algorithms`, and returns a seeded `torch.Generator`. `shared/` may be imported by any layer and may depend on torch/numpy (it already underpins color utilities). `synesthetic_config.py` is unchanged: `TrainingConfig.seed` already exists; this step finally consumes it.

## API Contracts

No changes. This step touches the training/evaluation path only; no FastAPI controllers or DTOs are added or modified. (Untouched layer for HTTP: No changes.)

## CLI Impact

`interface/cli/train.py`:

- New flag on `TrainArgs`: `deterministic: bool = False` (enables `torch.use_deterministic_algorithms(True)`).
- The configured `TrainingConfig.seed` is honoured at startup via the shared seeding helper.
- Best-checkpoint selection by structure-preservation score replaces unconditional save-last behaviour; the chosen correlation may be printed for P0-6 to capture.

No new CLI entry points; `encode.py`, `compare.py`, `compress.py`, `eval.py`, `query.py` are unchanged.

## Dependency Injection

- CLI wiring (factory-function style, matching the existing `_create_color_mapper` / `_create_dataset_adapter` pattern in `train.py`): a factory constructs `SpearmanStructurePreservationEvaluator` and injects it, together with the existing `ColorMapper` and `FileColorCodebookRepository`, into `TrainColorMappingUseCase` via its constructor.
- The use case depends on the `StructurePreservationEvaluator` abstraction (domain port), never the concrete scipy implementation.
- API Lagom `Container()` in `interface/api/main.py` only needs a binding if the evaluator is ever surfaced over HTTP; not required for this step.
- No new dependency is added; scipy is already declared in `setup.cfg` / `pyproject.toml`. `pip-audit` (tox gate) remains green because no new package is introduced.

## Observability

- Log (structured, with `correlation-id`) the resolved seed and the strict-determinism flag at training startup.
- Log the per-checkpoint structure-preservation score and which epoch/checkpoint was selected as best.
- Emit the structure-preservation correlation as a metric (gauge) so trends across runs are trackable.
- Existing per-epoch loss prints in the mappers are retained.

## Open Questions

1. **Where does best-checkpoint selection live without leaking torch into application?** Options: (a) extend the `ColorMapper` port with a per-epoch evaluation/save hook driven by the use case; (b) the use case scores a small set of post-training candidate checkpoints; (c) keep selection inside the mapper but inject the `StructurePreservationEvaluator` port into it. Preference: keep orchestration in `TrainColorMappingUseCase` to honour the layer rules.
2. **Source of held-out evaluation pairs for the metric:** a reserved slice of the already-encoded training embeddings, or a re-encoded eval split from the dataset adapter? Affects `train.py` wiring and determinism of the split (must itself be seeded — coordinate with `003-p0-3-shuffle-stratify-sampling`).
3. **`torch.use_deterministic_algorithms(True)` scope:** does it require `CUBLAS_WORKSPACE_CONFIG` / `set_num_threads(1)` to be bit-exact on the target device, and should the flag warn-or-error mode be configurable? Note `004-p0-4-eval-mapper-coverage-ef-fix` already proposes `set_num_threads(1)` for deterministic HNSW builds — reconcile so seeding is applied consistently.
4. **Color-space distance definition in the metric:** plain Euclidean over `(l, a, b)` (Delta E 76) versus reusing a shared Lab Delta-E utility in `shared/lab_utils.py`; and whether the metric should consume a `DistanceCalculator` instead of raw Lab distance for consistency with P0-1.
5. **Subsampling pairs for large batches:** for `n` Lab colors the pair count is `O(n^2)`; should the evaluator cap pairs via a seeded subsample, and is that cap a config field?
6. **Should the seeding helper live in `shared/` (preferred, importable by mappers and CLI) or be duplicated per mapper constructor?** A shared helper keeps all randomness sources seeded in one place and avoids divergence across the three mappers.
7. **Does the supervised mapper still need `torch.rand` targets after P0-2 deletes `_generate_targets`?** If P0-2 removes random-target generation, the `torch.rand` seeding obligation shrinks to weight init + `randperm`; sequence this step against P0-2 to avoid seeding code that P0-2 deletes.
