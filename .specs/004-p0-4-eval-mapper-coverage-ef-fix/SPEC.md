# Feature: Eval reaches every mapper variant and candidate retrieval honours ef >= k

## Overview

The `eval` CLI can today only evaluate the unconstrained `PyTorchColorMapper`. `interface/cli/eval.py` (`_create_color_classifier`, line ~73) hardcodes `PyTorchColorMapper` and `EvalArgs` exposes no `--mapper-type`. Pointing it at a checkpoint trained by the `structured` or `supervised` mapper (the variants produced by step `002-p0-2-structure-preserving-training`) calls `load_weights`, which calls `load_state_dict` against the wrong module tree: `PyTorchColorMapper` and `SupervisedPyTorchColorMapper` wrap `LabProjectorNetwork` (state-dict keys under `network.network.*`), whereas `StructuredPyTorchColorMapper` wraps `StructuredLabProjectorNetwork` (keys `backbone.*`, `lightness_head.*`, `hue_head.*`, `chroma_head.*`). The mismatch raises `RuntimeError` at load time, so the structure-preserving mappers from `002` are unevaluable end-to-end.

Separately, candidate retrieval in `infrastructure/evaluation/color_histogram_classifier.py` builds the hnswlib index with `ef=50` (default) but issues `knn_query(k=effective_candidates)` with `num_candidates=100`. hnswlib documents that the search-time `ef` must be `>= k`; `ef=50 < 100` yields degraded, under-filled candidate lists and therefore a silently weakened two-phase retrieval. `infrastructure/evaluation/hnsw_classifier.py` shares the same `ef=50` default and, while its default `k=5` does not currently breach `ef >= k`, it offers no guard and does not pin thread count, so both classifiers can return non-deterministic graphs across runs.

This feature (a) adds `--mapper-type` to `eval` and routes mapper construction through a single shared infrastructure factory reused by `train`, so every trained variant is evaluable without a state-dict error; and (b) enforces `ef = max(ef, num_candidates)` (and `ef = max(ef, k)` where there is no candidate stage) plus `hnswlib.set_num_threads(1)` for deterministic, fully-filled retrieval in both classifiers.

This step depends on `002-p0-2-structure-preserving-training`, which defines the structured and supervised mapper variants that are worth evaluating; without `002` there is no non-random checkpoint for `--mapper-type structured` to load. It feeds `006-p0-6-end-to-end-agnews-table`, which runs `train -> encode -> eval` to fill the AG News results table.

## User Stories

- As a researcher, I want `eval --mapper-type structured` to load a structured checkpoint and run without a state-dict error so that I can measure the structure-preserving mapper from step 002 on AG News.
- As a researcher, I want `eval --mapper-type supervised` to load a supervised checkpoint and run so that the supervised variant is evaluable on the same footing.
- As a researcher, I want candidate retrieval to satisfy hnswlib's `ef >= k` requirement so that the color-histogram k-NN reports honest accuracy rather than silently under-filled results.
- As a researcher, I want deterministic index builds so that repeated eval runs of the same checkpoint reproduce the same numbers.
- As a maintainer, I want mapper construction defined once and shared by `train` and `eval` so that the two CLIs cannot drift in how they build a mapper from a `mapper_type` and config.

## Acceptance Criteria

- [ ] Given a checkpoint saved by the structured mapper, when `main` runs with `EvalArgs(method="color", mapper_type="structured")`, then a `StructuredPyTorchColorMapper` is constructed and `load_weights` is called without a state-dict `RuntimeError`.
- [ ] Given a checkpoint saved by the supervised mapper, when `main` runs with `EvalArgs(method="color", mapper_type="supervised")`, then a `SupervisedPyTorchColorMapper` is constructed and loaded without error.
- [ ] Given default `EvalArgs(method="color")`, when the classifier is created, then `mapper_type` defaults to `"unconstrained"` and a `PyTorchColorMapper` is constructed (current behaviour preserved).
- [ ] Given `EvalArgs(mapper_type="structured")` and a config whose `structured_mapper` is `None`, when the mapper is created, then a `ValueError` is raised.
- [ ] Given a fitted `ColorHistogramClassifier` with `ef=50` and `num_candidates=100`, when `fit` completes, then `set_ef` is called with a value `>= num_candidates`.
- [ ] Given a fitted `ColorHistogramClassifier`, when `fit` runs, then `hnswlib.set_num_threads(1)` is invoked before the index is built.
- [ ] Given a fitted `HNSWClassifier` whose configured `ef < k`, when `fit` completes, then `set_ef` is called with a value `>= k`.
- [ ] Given a fitted `HNSWClassifier`, when `fit` runs, then `hnswlib.set_num_threads(1)` is invoked before the index is built.
- [ ] Given the shared mapper factory, when an unrecognised `mapper_type` is supplied, then a `ValueError` is raised.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. The factory returns the existing `domain/service/color_mapper.py::ColorMapper` abstraction; no new ports are required.

### Application Layer (`src/colors_of_meaning/application/`)

No changes. `application/use_case/evaluate_use_case.py` is listed in the roadmap because it is part of the eval path, but it already depends only on the abstract `Classifier`, `MetricsCalculator`, and `DatasetRepository`; mapper selection and the `ef >= k` fix sit below it, so its source is unchanged. (Its existing tests remain green and continue to act as the application-level guard.)

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- New `infrastructure/ml/color_mapper_factory.py` exposing `create_color_mapper(mapper_type: str, config: SynestheticConfig) -> ColorMapper`. It builds `PyTorchColorMapper` (`"unconstrained"`), `StructuredPyTorchColorMapper` (`"structured"`), or `SupervisedPyTorchColorMapper` (`"supervised"`), reading `ProjectorConfig` dims, `TrainingConfig.device`, and the `structured_mapper` / `supervised_mapper` sub-configs; it raises `ValueError` for unknown types and when a required sub-config is `None`. This is the extracted, signature-adjusted body of `interface/cli/train.py::_create_color_mapper`. Placing it in infrastructure mirrors the existing `infrastructure/system/health_factory.py` precedent (a module-level factory returning a domain abstraction) and keeps the interface layer free of concrete-mapper construction.
- `infrastructure/evaluation/color_histogram_classifier.py`: in `fit`, call `hnswlib.set_num_threads(1)` before building the index, and set the search parameter to `max(self.ef, self.num_candidates)` rather than the raw `self.ef`.
- `infrastructure/evaluation/hnsw_classifier.py`: in `fit`, call `hnswlib.set_num_threads(1)` before building the index, and set the search parameter to `max(self.ef, self.k)`. The existing module-level and method docstrings are removed to satisfy the no-comments rule (touched file).

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval.py`: add `mapper_type: str = "unconstrained"` to `EvalArgs`; replace the hardcoded `PyTorchColorMapper(...)` block in `_create_color_classifier` with a call to `create_color_mapper(args.mapper_type, config)`; import the factory and drop the now-unused direct `PyTorchColorMapper` import.
- `interface/cli/train.py`: replace the body of `_create_color_mapper` with a delegation to the shared `create_color_mapper(args.mapper_type, config)` so train and eval share one construction path. The `_create_color_mapper(args, config)` wrapper keeps its current signature (its tests pass `TrainArgs`), delegating internally.

### Shared Layer

No changes. `shared/synesthetic_config.py` already provides `ProjectorConfig`, `TrainingConfig.device`, `StructuredMapperConfig`, and `SupervisedMapperConfig`; the factory consumes them as-is.

## API Contracts

No HTTP API changes. This step touches only CLI and infrastructure; the FastAPI surface in `interface/api/` is untouched. No new or changed Pydantic DTOs.

## CLI Impact

- `eval` gains `--mapper-type {unconstrained,structured,supervised}`, default `unconstrained` (parsed by tyro from `EvalArgs`). Existing invocations without the flag behave exactly as before.
- `train`'s `--mapper-type` flag is unchanged in surface; only its internal construction path is centralised.

## Dependency Injection

Both CLIs continue to wire dependencies via factory functions plus constructor injection (the established CLI pattern; the API's Lagom `Container()` in `interface/api/main.py` is not involved). The new `create_color_mapper` factory becomes the single construction seam: `eval`'s `_create_color_classifier` and `train`'s `_create_color_mapper` both call it and inject the returned `ColorMapper` into `QuantizedColorMapper` / `TrainColorMappingUseCase` respectively. No new third-party dependency is introduced (hnswlib and the mapper classes already ship), so `setup.cfg` / `pyproject.toml` and `pip-audit` are unaffected.

## Observability

Construction and retrieval are surfaced through the existing CLI `print` statements (e.g. eval already prints the classifier choice). The factory does not introduce new logging; the determinism change (`set_num_threads(1)`, fixed `random_seed=100`) is itself an observability improvement because identical checkpoints now yield reproducible eval output. No metrics or tracing changes are in scope for this step.

## Open Questions

- Factory placement: this spec recommends `infrastructure/ml/color_mapper_factory.py` (mirrors `infrastructure/system/health_factory.py`; interface may import infrastructure). An alternative is `interface/cli/_mapper_factory.py` shared between the two CLI modules; it is equally layer-legal but treats the factory as interface-only glue and would not be reusable by future non-CLI callers. A third option is an application-layer factory, which is rejected because constructing concrete infrastructure mappers from inside `application/` would breach the application-imports-domain-only boundary.
- `EvalArgs.mapper_type` is typed `str` for parity with `TrainArgs.mapper_type` (also `str`); a `Literal["unconstrained","structured","supervised"]` would give tyro validation and match `EvalArgs.method`/`dataset`, but diverges from `train`. Should both CLIs migrate to `Literal` together (possibly under `014-p2-4-unify-config-systems`) rather than only `eval` here?
- The `set_ef` value: this spec uses `max(ef, num_candidates)`. If a future change lets `k > num_candidates`, the safe bound becomes `max(ef, num_candidates, k)`; out of scope now since `k (5) <= num_candidates (100)` holds, but noted so the guard is not under-specified later.
