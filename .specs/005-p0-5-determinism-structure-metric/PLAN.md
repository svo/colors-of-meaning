# Plan: determinism-and-structure-preservation-metric

## Implementation Strategy

Deliver two coupled capabilities behind clean hexagonal boundaries:

1. **Determinism.** Consume the already-defined `TrainingConfig.seed` (`shared/synesthetic_config.py`), which is currently never applied. Add one shared seeding helper in `shared/`, call it once at the CLI bootstrap (`interface/cli/train.py`), and thread a seeded `torch.Generator` into every stochastic call in the three mappers (`torch.randperm`, `torch.rand`) plus seed weight initialisation. Gate `torch.use_deterministic_algorithms(True)` behind a new CLI flag.

2. **Structure-preservation metric.** Introduce a new `domain/service` port `StructurePreservationEvaluator` (ABC, pure) and a `scipy.stats.spearmanr` implementation in `infrastructure/evaluation/`. Wire it into `TrainColorMappingUseCase` to drive best-checkpoint selection instead of unconditionally saving the last epoch.

Build TDD, domain-first. The metric directly measures the objective introduced by `002-p0-2-structure-preserving-training`; keep the port signature stable so P0-2 and P0-6 (end-to-end AG News table) can consume the same correlation. Scipy is already a declared dependency, so no `setup.cfg` / `pyproject.toml` change and no new `pip-audit` surface.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- New file `domain/service/structure_preservation_evaluator.py`: `StructurePreservationEvaluator(ABC)` mirroring `domain/service/distance_calculator.py`. Abstract methods: `evaluate(self, embeddings: npt.NDArray, lab_colors: List[LabColor]) -> float` and `metric_name(self) -> str`, each raising `NotImplementedError`. Imports limited to `abc`, `typing`, `numpy.typing`, and `domain.model.lab_color.LabColor`. No torch / scipy / sklearn.
- `domain/model/lab_color.py` and `domain/model/colored_document.py` are inputs only — No changes.

### Application Layer (`src/colors_of_meaning/application/`)

- Edit `application/use_case/train_color_mapping_use_case.py`: add `structure_preservation_evaluator: StructurePreservationEvaluator` to `__init__`; extend `execute(...)` to accept held-out evaluation embeddings (and `seed`) and to persist the best-scoring checkpoint via the injected evaluator rather than unconditionally calling `save_weights`. Imports stay domain-only.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- New file `infrastructure/evaluation/structure_preservation_evaluator.py`: `SpearmanStructurePreservationEvaluator(StructurePreservationEvaluator)` using `scipy.stats.spearmanr` (pattern of `infrastructure/ml/wasserstein_distance_calculator.py`). Builds the upper-triangle embedding cosine-similarity vector and the matching Lab-distance vector (Euclidean over `LabColor`), returns the correlation as `float`, raises `ValueError` for fewer than two pairs. Optional seeded subsampling for large pair counts.
- Edit `infrastructure/ml/pytorch_color_mapper.py`, `structured_pytorch_color_mapper.py`, `supervised_pytorch_color_mapper.py`: seed weight init; create a seeded `torch.Generator` on `self.device`; pass `generator=` to `torch.randperm` (`pytorch` L94, `structured` L139, `supervised` L132) and `torch.rand` (`pytorch` L128-130, `supervised` L204-206). Source the seed via the shared helper / constructor argument.

### Interface Layer (`src/colors_of_meaning/interface/`)

- Edit `interface/cli/train.py`: add `deterministic: bool = False` to `TrainArgs`; call the shared seeding helper once in `main` before constructing mappers / encoding; construct `SpearmanStructurePreservationEvaluator` and inject it into `TrainColorMappingUseCase` in `_execute_training`; supply held-out evaluation embeddings; optionally print the selected correlation.
- API controllers / DTOs: No changes.

### Shared Layer (`src/colors_of_meaning/shared/`)

- New file `shared/determinism.py`: `seed_everything(seed: int, deterministic: bool = False) -> torch.Generator` wrapping `torch.manual_seed`, `numpy.random.seed`, optional `torch.use_deterministic_algorithms(True)`, returning a seeded generator.
- `shared/synesthetic_config.py`: No changes (`TrainingConfig.seed` already exists and is now consumed).

## Dependency Injection

- CLI factory style (matching `_create_color_mapper` / `_create_dataset_adapter`): build `SpearmanStructurePreservationEvaluator` and pass it, with the existing `ColorMapper` and `FileColorCodebookRepository`, into `TrainColorMappingUseCase.__init__`.
- The use case depends only on the `StructurePreservationEvaluator` abstraction.
- API Lagom `Container()` (`interface/api/main.py`): no binding required for this step.
- No new third-party dependency; scipy already declared. `pip-audit` (tox) stays green.

## Task List

1. [ ] domain: add `StructurePreservationEvaluator` ABC in `domain/service/structure_preservation_evaluator.py` (`evaluate`, `metric_name`), pure imports only.
2. [ ] domain: ensure `tests/colors_of_meaning/domain/service/__init__.py` covers the new port with a NotImplementedError contract test.
3. [ ] shared: add `seed_everything(seed, deterministic=False) -> torch.Generator` in `shared/determinism.py`.
4. [ ] infrastructure: seed weight init and thread a seeded `torch.Generator` into `torch.randperm` / `torch.rand` in `pytorch_color_mapper.py`.
5. [ ] infrastructure: apply the same seeding to `structured_pytorch_color_mapper.py` (`randperm` L139).
6. [ ] infrastructure: apply the same seeding to `supervised_pytorch_color_mapper.py` (`randperm` L132, `torch.rand` L204-206), sequenced against P0-2 which may delete `_generate_targets`.
7. [ ] infrastructure: implement `SpearmanStructurePreservationEvaluator` in `infrastructure/evaluation/structure_preservation_evaluator.py` using `scipy.stats.spearmanr` over upper-triangle pairs; raise `ValueError` for fewer than two pairs.
8. [ ] application: inject `StructurePreservationEvaluator` into `TrainColorMappingUseCase`; extend `execute` to score checkpoints and persist the best-scoring weights.
9. [ ] interface: add `deterministic` flag to `TrainArgs`; call `seed_everything` once at bootstrap in `interface/cli/train.py`.
10. [ ] interface: construct and inject `SpearmanStructurePreservationEvaluator`, and supply held-out evaluation embeddings, in `_execute_training`.
11. [ ] tests: domain port contract test.
12. [ ] tests: `seed_everything` returns a generator and applies seeds / honours the deterministic flag.
13. [ ] tests: evaluator returns a correlation in `[-1, 1]`.
14. [ ] tests: perfectly inverse-ranked similarity vs distance returns `-1.0` within tolerance.
15. [ ] tests: rank-unrelated inputs return a near-zero magnitude correlation.
16. [ ] tests: evaluator raises `ValueError` for fewer than two pairs.
17. [ ] tests: identical seed reproduces identical Lab outputs for the same input embedding (per mapper).
18. [ ] tests: `TrainColorMappingUseCase` persists the best-scoring checkpoint (mock `ColorMapper` + stub evaluator).
19. [ ] tests: CLI `deterministic` flag triggers `torch.use_deterministic_algorithms` (patched).
20. [ ] tests: pytest-archon rule confirms the metric stays out of domain (scipy import only in infrastructure).
21. [ ] verify: run `tox` (8 gates) for 100% coverage, flake8, black, bandit, semgrep, mypy, xenon, pip-audit.

## Testing Strategy

- One logical assertion per test; ML/numerical tests may group related asserts on the same result (e.g. shape plus range). Names follow `test_should_<behaviour>_when_<condition>`.
- `assertpy` (`assert_that`) for entity-style tests; plain `assert` / `pytest.raises` for ML and domain-specific tests (e.g. evaluator correlation, seed reproduction, `ValueError` paths).
- Determinism tests construct the same mapper twice with the same seed and assert identical Lab outputs from `embed_to_lab` (no network calls; synthetic numpy embeddings).
- Evaluator tests use hand-built embeddings and `LabColor` lists with known monotonic relationships to assert `-1.0`, near-zero, and the `[-1, 1]` bound.
- Best-checkpoint test injects a mock `ColorMapper` and a stub `StructurePreservationEvaluator` returning scripted scores; asserts the highest-scoring weights are persisted.
- Architecture: extend `tests/colors_of_meaning/test_synesthetic_architecture.py` (pytest-archon) so `domain.*` still imports no infrastructure and the evaluator's scipy dependency lives only in `infrastructure.evaluation`.
- New tests mirror source under `tests/colors_of_meaning/domain/service/`, `tests/colors_of_meaning/infrastructure/evaluation/`, `tests/colors_of_meaning/infrastructure/ml/`, `tests/colors_of_meaning/application/use_case/`, `tests/colors_of_meaning/interface/cli/`, and `tests/colors_of_meaning/shared/`, each with `__init__.py`.
- No comments in test or source. Final verification via `tox` only (never bare pytest); 100% coverage required.

## Observability Plan

- Structured log (with `correlation-id`) of the resolved seed and `deterministic` flag at training startup.
- Log each checkpoint's structure-preservation score and the selected best epoch/checkpoint.
- Emit the structure-preservation correlation as a gauge metric for cross-run tracking.
- Retain existing per-epoch loss prints in the mappers.

## Risks and Mitigations

- **Bit-exact determinism may need device/threading controls.** `torch.use_deterministic_algorithms(True)` can require `CUBLAS_WORKSPACE_CONFIG` and single-threaded ops. Mitigation: gate behind the `deterministic` flag, document the requirement, and reconcile with `004-p0-4-eval-mapper-coverage-ef-fix` (`set_num_threads(1)`); default off so normal runs stay fast.
- **Layer leakage.** Injecting the evaluator for per-epoch scoring risks pulling torch into application. Mitigation: keep torch out of `TrainColorMappingUseCase`; expose checkpoint scoring through the domain `ColorMapper`/evaluator ports — resolve via Open Question 1 before coding task 8.
- **Sequencing with P0-2.** `002-p0-2-structure-preserving-training` may delete `_generate_targets`, removing some `torch.rand` calls this step seeds. Mitigation: land seeding for `randperm` and weight init first; seed `torch.rand` only where targets remain after P0-2 (Open Question 7).
- **O(n^2) pair cost.** Large batches make full-pairwise Spearman expensive. Mitigation: seeded subsampling cap (Open Question 5), itself reproducible via the seeded generator.
- **Held-out split determinism.** The metric's evaluation pairs must come from a seeded, reproducible split. Mitigation: coordinate with `003-p0-3-shuffle-stratify-sampling` so the split seed is threaded consistently (Open Question 2).
