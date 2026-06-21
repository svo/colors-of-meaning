# Plan: eval-mapper-coverage-ef-fix

## Implementation Strategy

Two independent fixes delivered in one branch, each driven test-first.

1. Mapper coverage. Extract the mapper construction in `interface/cli/train.py::_create_color_mapper` into a new infrastructure factory `infrastructure/ml/color_mapper_factory.py::create_color_mapper(mapper_type, config)` that takes a `mapper_type: str` (not the `TrainArgs` dataclass) so both CLIs can call it. Re-point `train`'s `_create_color_mapper` at it (keeping its `(args, config)` signature for its existing tests). Add `mapper_type` to `EvalArgs` and route `eval`'s `_create_color_classifier` through the factory, removing the hardcoded `PyTorchColorMapper`. This makes structured/supervised checkpoints loadable because the matching network (and therefore the matching state-dict key tree) is constructed before `load_weights`.

2. Retrieval correctness and determinism. In both `color_histogram_classifier.py` and `hnsw_classifier.py`, call `hnswlib.set_num_threads(1)` before index construction and raise the search parameter to `max(ef, num_candidates)` (color histogram, which has a candidate stage) / `max(ef, k)` (hnsw, which queries `k` directly).

Sequence the work domain -> application -> infrastructure -> interface -> tests. Domain and application need no source edits (the factory returns the existing `ColorMapper` port; `EvaluateUseCase` already depends on abstractions only), so the first code change is the infrastructure factory. Existing tests that pin current behaviour (`set_ef(50)`, `eval.PyTorchColorMapper` patch targets) are updated as part of the relevant step. Verify with `tox` (all 8 gates) at the end; never `pytest` alone.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. `domain/service/color_mapper.py::ColorMapper` is the return type of the factory and already covers all three variants.

### Application Layer (`src/colors_of_meaning/application/`)

No changes. `application/use_case/evaluate_use_case.py` depends only on `Classifier`, `MetricsCalculator`, `DatasetRepository`; mapper selection and `ef` are below it. Its existing tests remain the application-level guard.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- New `infrastructure/ml/color_mapper_factory.py`:
  - `create_color_mapper(mapper_type: str, config: SynestheticConfig) -> ColorMapper`.
  - `"structured"` -> `StructuredPyTorchColorMapper(input_dim=config.projector.embedding_dim, hidden_dim_1=..., hidden_dim_2=..., dropout_rate=..., device=config.training.device, alpha/beta/gamma/num_clusters/max_chroma=config.structured_mapper.*)`; raise `ValueError` if `config.structured_mapper is None`.
  - `"supervised"` -> `SupervisedPyTorchColorMapper(..., num_classes=config.supervised_mapper.num_classes, classification_weight=config.supervised_mapper.classification_weight)`; raise `ValueError` if `config.supervised_mapper is None`.
  - default/`"unconstrained"` -> `PyTorchColorMapper(...)`; unrecognised `mapper_type` -> `ValueError`.
  - No comments; mapping expressed via small explicit branches matching the existing `train.py` structure.
- `infrastructure/evaluation/color_histogram_classifier.py::fit`: add `hnswlib.set_num_threads(1)` before `hnswlib.Index(...)`; change `self.index.set_ef(self.ef)` to `self.index.set_ef(max(self.ef, self.num_candidates))`.
- `infrastructure/evaluation/hnsw_classifier.py::fit`: add `hnswlib.set_num_threads(1)` before `hnswlib.Index(...)`; change `self.index.set_ef(self.ef)` to `self.index.set_ef(max(self.ef, self.k))`. Remove the module-level class docstring, the `__init__` docstring, and the inline `# Create...`, `# Add...`, `# Set...` comments (no-comments rule applies to the touched file).

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/eval.py`:
  - Add `mapper_type: str = "unconstrained"` to `EvalArgs`.
  - Import `from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper`.
  - In `_create_color_classifier`, replace the `PyTorchColorMapper(...)` construction with `color_mapper = create_color_mapper(args.mapper_type, config)`; keep the subsequent `load_weights`, codebook load, `QuantizedColorMapper`, `EncodeDocumentUseCase`, `ColorHistogramClassifier` wiring.
  - Remove the now-unused `PyTorchColorMapper` import.
- `interface/cli/train.py`:
  - Replace the body of `_create_color_mapper(args, config)` with `return create_color_mapper(args.mapper_type, config)`; import the factory. Keep the wrapper so `tests/.../cli/test_train.py`, which calls `_create_color_mapper(args, config)`, stays valid. (The direct `StructuredPyTorchColorMapper` / `SupervisedPyTorchColorMapper` / `PyTorchColorMapper` imports become unused in `train.py` and are removed; they live in the factory now.)

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes. `synesthetic_config.py` already provides every field the factory reads.

## Dependency Injection

The factory is the single construction seam, consumed by both CLI factory functions (the established CLI DI pattern; no Lagom). `create_color_mapper` returns the `ColorMapper` abstraction; `eval` injects it into `QuantizedColorMapper` and `train` into `TrainColorMappingUseCase`. No new dependency is added, so `setup.cfg`, `pyproject.toml`, and `pip-audit` are unchanged.

## Task List

1. [ ] domain: confirm no change needed; rely on `domain/service/color_mapper.py::ColorMapper` as the factory return type.
2. [ ] application: confirm no change needed; rely on existing `evaluate_use_case.py` tests as the application guard.
3. [ ] infrastructure: write failing tests for `infrastructure/ml/color_mapper_factory.py::create_color_mapper` (unconstrained/structured/supervised types, both `None`-sub-config `ValueError`s, unknown-type `ValueError`) under `tests/colors_of_meaning/infrastructure/ml/test_color_mapper_factory.py`.
4. [ ] infrastructure: implement `create_color_mapper` by extracting and signature-adjusting `train.py::_create_color_mapper` (`mapper_type: str` instead of `TrainArgs`); make the tests pass.
5. [ ] infrastructure: update `tests/.../evaluation/test_color_histogram_classifier.py::test_should_build_hnsw_index_with_cosine_space` (currently asserts `set_ef(50)`) to expect `set_ef(100)`; add tests asserting `set_ef` receives `>= num_candidates` and that `hnswlib.set_num_threads(1)` is called in `fit`.
6. [ ] infrastructure: implement the `color_histogram_classifier.py::fit` changes (`set_num_threads(1)`, `set_ef(max(ef, num_candidates))`); make the tests pass.
7. [ ] infrastructure: add tests to `tests/.../evaluation/test_hnsw_classifier.py` asserting `set_ef` receives `>= k` and `hnswlib.set_num_threads(1)` is called in `fit`.
8. [ ] infrastructure: implement the `hnsw_classifier.py::fit` changes and remove its docstrings/comments; make the tests pass.
9. [ ] interface: update `tests/.../cli/test_train.py` so `_create_color_mapper` tests still pass against the delegating wrapper (no signature change expected; adjust patch targets only if construction moves to the factory module).
10. [ ] interface: re-point `train.py::_create_color_mapper` at the factory; remove now-unused mapper imports.
11. [ ] interface: write failing tests for `eval.py` covering `EvalArgs` default `mapper_type == "unconstrained"`, and color-method runs with `mapper_type` `structured` and `supervised` constructing the matching mapper without a `load_weights` error (patching the factory and `load_weights`).
12. [ ] interface: add `mapper_type` to `EvalArgs`, route `_create_color_classifier` through `create_color_mapper`, remove the unused `PyTorchColorMapper` import; update the existing color-method tests (`test_should_execute_evaluation_with_color_method`, `test_should_raise_error_when_codebook_not_found`) so their patch targets follow the factory delegation.
13. [ ] tests: run `tox` and resolve any flake8/black/mypy/bandit/semgrep/xenon/radon/pip-audit findings; confirm 100% coverage including the factory's `ValueError` branches.

## Testing Strategy

- One logical assertion per test; ML/infra tests may group related asserts on the same result. Use plain `assert` / `pytest.raises` for these infrastructure and CLI tests (consistent with the existing `test_train.py`, `test_eval.py`, `test_color_histogram_classifier.py`); reserve `assertpy` for base-entity tests, of which this step has none.
- Names follow `test_should_<behaviour>_when_<condition>`, e.g. `test_should_construct_structured_mapper_when_mapper_type_is_structured`, `test_should_raise_value_error_when_mapper_type_is_unknown`, `test_should_set_ef_at_least_num_candidates_when_fitting`, `test_should_pin_single_thread_when_building_index`, `test_should_load_structured_checkpoint_without_state_dict_error_when_mapper_type_is_structured`.
- Mapper-coverage CLI tests mock the factory and `load_weights` (no real torch load, no network, no dataset I/O) so the structured/supervised acceptance is proven by "matching mapper constructed + `load_weights` called, no `RuntimeError`" rather than a real checkpoint round-trip; an optional infrastructure-level round-trip test (save a `StructuredLabProjectorNetwork` state-dict, `create_color_mapper("structured", config).load_weights(path)`) can additionally pin the real key-compatibility cheaply on CPU.
- Retrieval tests patch `hnswlib.Index` and `hnswlib.set_num_threads` (as the existing tests already patch `hnswlib.Index`) and assert on the recorded `set_ef` / `set_num_threads` calls.
- Architecture: the existing `pytest-archon` suites (`tests/colors_of_meaning/test_architecture.py`, `test_synesthetic_architecture.py`) already enforce layer boundaries; the new factory lives in infrastructure and imports only domain + infrastructure, so it satisfies them. Add a focused archrule only if a gap surfaces; do not duplicate existing rules.
- 100% coverage via `tox` (8 gates). Never run `pytest` alone for verification. Use `tox -- tests/colors_of_meaning/infrastructure/ml/test_color_mapper_factory.py` for fast iteration.

## Observability Plan

No new logging, metrics, or tracing. The determinism change (`set_num_threads(1)` plus the existing fixed `random_seed=100`) makes eval output reproducible for a given checkpoint, which is the observability-relevant outcome of this step. Existing CLI `print` lines (classifier choice, results) are retained.

## Risks and Mitigations

- Risk: a structured/supervised checkpoint trained before `002-p0-2-structure-preserving-training` lands may not exist. Mitigation: this step explicitly Depends on `002`; mapper-coverage tests use mocked `load_weights` so they do not require a real artifact, and `006-p0-6-end-to-end-agnews-table` exercises the real round-trip.
- Risk: existing tests pin old behaviour (`set_ef(50)`; `eval.PyTorchColorMapper` and `train` direct-mapper patch targets). Mitigation: tasks 5, 9, and 12 update those tests in lockstep with the source change so coverage and assertions stay truthful.
- Risk: removing now-unused imports in `eval.py` / `train.py` could break a test that patches them by module path (e.g. `eval.PyTorchColorMapper`). Mitigation: re-point those patches at `infrastructure.ml.color_mapper_factory` (or the factory call) when delegation moves construction out of the CLI module.
- Risk: `set_ef(max(ef, num_candidates))` interacts with `effective_candidates = min(num_candidates, len(training_labels))`; for tiny corpora `ef` may exceed element count. Mitigation: hnswlib tolerates `ef > max_elements` (it clamps internally); the small-corpus test (`test_should_handle_fewer_training_samples_than_num_candidates`) continues to assert `knn_query` k clamps to the element count, confirming behaviour.
- Risk: the no-comments rule trips xenon/flake8 only after docstrings are removed from `hnsw_classifier.py` if a method becomes a one-liner with a bare body. Mitigation: keep method bodies expressive; run `tox -e format` then `tox` before declaring done.
