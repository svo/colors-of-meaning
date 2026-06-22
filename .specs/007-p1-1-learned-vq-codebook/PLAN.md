# Plan: learned-vq-codebook

## Implementation Strategy
Add a learned vector-quantization palette without violating hexagonal boundaries: the k-means fit lives entirely in a new infrastructure adapter, the domain entity keeps accepting any `List[LabColor]`, the application use case chooses between uniform and learned via an injected domain-level abstraction, and the CLI wires the concrete factory through a factory function. In parallel, vectorise the domain `ColorCodebook.quantize` so encoding stops paying an O(num_bins) Python loop per color, keeping its result bit-for-bit equivalent to the current first-minimum semantics.

1. Domain: vectorise `ColorCodebook.quantize` (`domain/model/color_codebook.py` lines 19-29) to a single batched nearest-centroid lookup over a precomputed `(num_bins, 3)` numpy array of palette coordinates; preserve tie-breaking (`<`, first minimum). Keep `create_uniform_grid` verbatim as the A/B baseline. Define the codebook-factory abstraction (`ABC`) in `domain/` so the application can depend on it without importing infrastructure.
2. Application: turn the hardcoded `create_uniform_grid` at `train_color_mapping_use_case.py` line 32 into the default branch of a codebook-mode decision; inject the factory abstraction and add a mode parameter to `execute(...)`; delegate learned-palette construction to the factory using `self.color_mapper` and the `embeddings` already passed in.
3. Infrastructure: implement `infrastructure/ml/learned_color_codebook_factory.py` that projects embeddings via `color_mapper.embed_batch_to_lab(...)`, fits `MiniBatchKMeans`, clamps `cluster_centers_` into valid `LabColor`s, and returns a `ColorCodebook`. Reuse the in-repo sklearn pattern from `pq_compression_baseline.py`.
4. Interface: add a codebook-mode flag to `TrainArgs`, build the factory in `_execute_training`, inject it, and thread the mode into `use_case.execute(...)`; default stays `uniform`.
5. Tests: TDD throughout (write the failing test first), one logical assertion per test, plus a `pytest-archon` boundary test asserting `domain/` imports no sklearn.
6. Cross-reference `002-p0-2-structure-preserving-training` (supplies the meaningful Lab distribution the palette is fitted to) and `001-p0-1-lab-emd-distance` (benefits from bins that land on real density); the uniform-vs-learned reconstruction A/B report itself is `009-p1-3-real-compression-comparison`.

Develop test-first: add the failing factory test (`build` returns a `ColorCodebook` with `num_bins` centroid colors) against an absent module, confirm it fails, then implement.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
- `domain/model/color_codebook.py`
  - Rewrite `quantize` (lines 19-29): precompute (once, e.g. cached numpy array derived from `self.colors`) a `(num_bins, 3)` array of `[l, a, b]`; compute squared Euclidean distance from the query color to all rows and return `int(np.argmin(...))`. Match current first-minimum tie semantics so existing quantize tests pass unchanged.
  - Keep `get_color` (lines 31-34) and `create_uniform_grid` (lines 43-57) unchanged; retain or inline `_euclidean_distance` (lines 36-41) consistently with the vectorised form (remove only if it becomes dead and its test is migrated).
  - The entity remains a frozen dataclass over `List[LabColor]`, imports only numpy (already line 3) — no sklearn/torch.
- New codebook-factory abstraction in `domain/` (an `ABC`, e.g. `domain/service/color_codebook_factory.py`) declaring `build(embeddings, num_bins, seed) -> ColorCodebook`, so the application depends on the port, not the infrastructure implementation. (If the team prefers, the existing `ColorMapper`-style placement under `domain/service/` is the natural home.)
- `domain/model/lab_color.py`, `domain/repository/color_codebook_repository.py`, `domain/service/color_mapper.py`: unchanged; reused as-is (`LabColor.clamp`, the repository port, `embed_batch_to_lab`).

### Application Layer (`src/colors_of_meaning/application/`)
- `application/use_case/train_color_mapping_use_case.py`
  - Add the factory abstraction as an optional constructor-injected collaborator alongside `color_mapper` and `codebook_repository`.
  - Add a codebook-mode argument to `execute(...)` (e.g. `codebook_mode: str` defaulting to uniform).
  - Replace line 32 with a branch: uniform -> `ColorCodebook.create_uniform_grid(bins_per_dimension=bins_per_dimension)` (current behaviour); learned -> `self.codebook_factory.build(embeddings=embeddings, num_bins=..., seed=...)`. Then `self.codebook_repository.save(codebook, codebook_name)` unchanged.
  - Imports stay domain-only (the factory is a domain `ABC`); no infrastructure import. The k-means math does not enter this layer.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- New `infrastructure/ml/learned_color_codebook_factory.py`
  - Implements the domain factory `ABC`; constructor-injects the trained `ColorMapper`.
  - `build(embeddings, num_bins, seed)`: `lab_colors = self.color_mapper.embed_batch_to_lab(embeddings)`; stack to `points = np.array([[c.l, c.a, c.b] for c in lab_colors], dtype=np.float32)`; `effective_clusters = min(num_bins, len(unique points))` for the degenerate case; `kmeans = MiniBatchKMeans(n_clusters=effective_clusters, random_state=seed, n_init=..., batch_size=min(256, len(points)))`; `kmeans.fit(points)`; map each `cluster_centers_` row to a clamped `LabColor` (via `LabColor.from_tuple(...).clamp()` or `np.clip` to Lab bounds); pad/handle so the returned `ColorCodebook(colors=..., num_bins=...)` satisfies its `len(colors) == num_bins` invariant (final shape decided in Open Questions).
  - Mirror existing usage: `pq_compression_baseline.py` lines 44-50 (`MiniBatchKMeans(..., random_state=42, n_init=1, batch_size=min(256, num_samples))`, `.fit`, `.predict`, `.cluster_centers_`). Confirm `n_init` per Open Questions (ROADMAP says `'auto'`; repo precedent is `1`/`10`).
  - `import` sklearn only here; the factory performs no file I/O and never re-trains the mapper.
- `infrastructure/persistence/file_color_codebook_repository.py`: unchanged; pickles any `ColorCodebook`.

### Interface Layer (`src/colors_of_meaning/interface/`)
- `interface/cli/train.py`
  - `TrainArgs` (lines 36-42): add the codebook-mode flag (proposed `codebook_mode: str = "uniform"`), defaulting to current behaviour.
  - Add a factory function (e.g. `_create_codebook_factory(color_mapper)`) returning `LearnedColorCodebookFactory(color_mapper=color_mapper)`.
  - `_execute_training` (lines 145-166): build the factory, inject it into `TrainColorMappingUseCase(color_mapper=..., codebook_repository=..., codebook_factory=...)`, and pass `codebook_mode=args.codebook_mode` (and the `num_bins`/`seed` it needs) into `use_case.execute(...)`. Retain the existing print lines (164-165); optionally print the chosen mode.
- API (`interface/api/...`): unchanged.

### Shared Layer (`src/colors_of_meaning/shared/`)
- `shared/lab_utils.py`: reused for clamping (`LabColor.clamp`, or a numpy clamp against documented Lab bounds) and as the reference Euclidean Lab distance (`delta_e`). A batched clamp helper is added here only if the Open Question decides for it (numpy-only, no new dependency).
- `shared/synesthetic_config.py`: `CodebookConfig.num_bins` (line 19) feeds the factory's `n_clusters`; `TrainingConfig.seed` feeds `random_state`. A persisted `codebook.mode` field is added (with `from_yaml`/`to_yaml` round-trip tests at lines 86, 97) only if the Open Question selects config-persisted mode over CLI-only.

## Dependency Injection
No Lagom change. The CLI keeps wiring through factory functions and constructor injection: `TrainColorMappingUseCase` gains a third injected collaborator (`codebook_factory`) built by `_create_codebook_factory(color_mapper)` in `interface/cli/train.py`, exactly mirroring how `FileColorCodebookRepository()` is constructed and injected today (line 152). The application depends on the factory through the domain `ABC`, never importing infrastructure. The API Lagom `Container()` in `interface/api/main.py` is untouched (training is CLI-only). No direct instantiation of the factory inside the use case.

## Task List
1. [ ] domain: rewrite `ColorCodebook.quantize` (`color_codebook.py` lines 19-29) to a vectorised `np.argmin` nearest-centroid lookup preserving first-minimum tie semantics; keep `create_uniform_grid` as the baseline.
2. [ ] domain: define the codebook-factory `ABC` (`domain/service/color_codebook_factory.py`) with `build(embeddings, num_bins, seed) -> ColorCodebook`, importing only domain symbols.
3. [ ] application: extend `TrainColorMappingUseCase.__init__` to constructor-inject the factory abstraction and `execute(...)` to accept a codebook-mode argument.
4. [ ] application: replace the hardcoded `create_uniform_grid` at `train_color_mapping_use_case.py` line 32 with a uniform/learned branch that delegates the learned path to the injected factory, then saves via the existing repository port.
5. [ ] infrastructure: create `infrastructure/ml/learned_color_codebook_factory.py` implementing the `ABC`; project embeddings via injected `ColorMapper.embed_batch_to_lab`, fit `MiniBatchKMeans(n_clusters=num_bins, random_state=seed, ...)`, clamp `cluster_centers_` to valid `LabColor`s, return a `ColorCodebook`.
6. [ ] infrastructure: handle the degenerate fewer-unique-points-than-`num_bins` and empty-embeddings cases deterministically so the `len(colors) == num_bins` invariant holds.
7. [ ] interface: add the codebook-mode flag to `TrainArgs`, add `_create_codebook_factory`, and in `_execute_training` inject the factory and thread the mode (plus `num_bins`/`seed`) into `use_case.execute(...)`; keep the default uniform.
8. [ ] tests: write the domain quantize-equivalence tests (vectorised result matches old loop incl. tie-breaking) under `tests/colors_of_meaning/domain/model/test_color_codebook.py`.
9. [ ] tests: write the infrastructure factory tests (shape/`num_bins`, centroids from `cluster_centers_`, determinism, clamping, degenerate cases) under `tests/colors_of_meaning/infrastructure/ml/test_learned_color_codebook_factory.py`.
10. [ ] tests: write the application use-case tests (uniform default path; learned path delegates to a mocked factory and saves the returned codebook).
11. [ ] tests: write the interface test that the train CLI builds and injects the factory and selects mode correctly (mode default = uniform).
12. [ ] tests: write/confirm the `pytest-archon` boundary tests (domain imports no sklearn/torch; application imports no infrastructure).
13. [ ] tests: run `tox` for full verification (all 8 gates) and confirm 100% coverage; never rely on `pytest` alone.

## Testing Strategy
Conventions: one logical assertion per test (ML/numerical tests may group related asserts on a single result, e.g. shape plus value range plus `num_bins`); names `test_should_<behaviour>_when_<condition>`; ML/infrastructure/domain tests use plain `assert`/`pytest.raises` (matching the existing mapper/codebook tests); base-entity assertions elsewhere use `assertpy` (`assert_that`); synthetic numpy data and mocked `ColorMapper`/factory only, no network or dataset downloads. Use a fixed seed for determinism assertions.

New/updated tests:

Domain (`tests/colors_of_meaning/domain/model/test_color_codebook.py`):
- `test_should_return_nearest_bin_index_when_quantizing_color` — vectorised `quantize` returns the index of the nearest palette color for a known small palette.
- `test_should_match_first_minimum_when_distances_tie` — equidistant palette colors resolve to the first index, matching the old `<` semantics.
- `test_should_quantize_identically_to_uniform_grid_baseline_when_color_in_range` — vectorised result equals an independent argmin reference over `create_uniform_grid` colors.
- Preserve existing `create_uniform_grid` shape/`num_bins` and `get_color` range tests.

Infrastructure (`tests/colors_of_meaning/infrastructure/ml/test_learned_color_codebook_factory.py`):
- `test_should_return_codebook_with_num_bins_colors_when_built` — `build` returns a `ColorCodebook` whose `num_bins` and `len(colors)` equal the requested bins (related asserts on one result).
- `test_should_use_cluster_centers_as_palette_when_fitting_kmeans` — palette colors correspond to `MiniBatchKMeans.cluster_centers_` (assert via a patched/seeded estimator or by recomputing centroids on a tiny synthetic set).
- `test_should_fit_on_projected_lab_colors_when_building` — the array passed to k-means comes from `color_mapper.embed_batch_to_lab(...)` (assert the mapper port is invoked with the embeddings, not raw 384-dim fed to a 3-D fit).
- `test_should_produce_identical_palette_when_seed_is_fixed` — two `build` calls with the same seed/inputs yield identical palettes.
- `test_should_clamp_centroid_when_out_of_lab_range` — an out-of-gamut centroid is clamped so `LabColor` construction does not raise.
- `test_should_reduce_clusters_when_unique_points_fewer_than_num_bins` — degenerate case yields a valid codebook satisfying the `len(colors) == num_bins` invariant (per chosen Open-Question behaviour).
- `test_should_set_n_clusters_to_num_bins_and_random_state_to_seed_when_constructing_kmeans` — estimator is configured with the expected `n_clusters`/`random_state`.

Application (`tests/colors_of_meaning/application/use_case/test_train_color_mapping_use_case.py`):
- `test_should_build_uniform_codebook_when_mode_is_uniform` — default path calls `create_uniform_grid` and saves it (factory not invoked).
- `test_should_delegate_to_factory_when_mode_is_learned` — learned path calls the injected (mocked) factory's `build` and saves the returned codebook.
- `test_should_save_codebook_under_given_name_when_executing` — repository `save` is called with the produced codebook and `codebook_name`.

Interface (`tests/colors_of_meaning/interface/cli/test_train.py`):
- `test_should_default_codebook_mode_to_uniform_when_not_specified` — `TrainArgs` default preserves current behaviour.
- `test_should_inject_learned_codebook_factory_into_use_case_when_training` — `_execute_training` constructs and injects the factory.

Architectural / boundary tests (`pytest-archon`):
- `test_should_not_import_sklearn_in_domain_layer` — `domain/` is free of sklearn (and torch); k-means is confined to `infrastructure/ml/`.
- `test_should_not_import_infrastructure_in_application_layer` — the use case depends only on the domain factory `ABC`, not the concrete factory.

Cross-referenced acceptance (delivered/asserted elsewhere): the uniform-versus-learned reconstruction A/B comparison and the "~88% of bins reclaimed" reconstruction-error improvement are owned by `009-p1-3-real-compression-comparison`; the meaningful Lab distribution the palette is fitted to is delivered by `002-p0-2-structure-preserving-training`; improved Wasserstein retrieval over learned bins is validated by `001-p0-1-lab-emd-distance`. This plan's factory/quantize tests are the local proxies.

Final verification: `tox` (flake8, black, bandit, semgrep, xenon, radon, mypy, pip-audit) green with 100% coverage. For fast TDD iteration use `tox -- tests/colors_of_meaning/infrastructure/ml/test_learned_color_codebook_factory.py`, never bare `pytest`.

## Observability Plan
Keep the existing every-run `print` lines in `train.py` (lines 164-165) and add a structured JSON log (with `correlation-id`) recording codebook mode, `num_bins`, number of projected colors fed to k-means, and the learned palette's inertia / mean reconstruction error so the bins-reclaimed claim is measurable. Emit a fit-duration metric and a reconstruction-error gauge. Name helpers so intent is self-evident (no comments). The vectorised `quantize` adds no logging and must not regress existing output. No secrets logged.

## Risks and Mitigations
- Boundary violation (sklearn leaking into domain or infrastructure leaking into application): mitigate by keeping the k-means fit only in `infrastructure/ml/learned_color_codebook_factory.py`, depending on a domain `ABC`, and enforcing both with `pytest-archon` boundary tests (Tasks 12).
- Behavioural drift from vectorising `quantize` (tie-breaking changes break callers/tests): mitigate by preserving first-minimum (`<`) semantics and asserting equivalence to an independent argmin reference (domain tests), so existing histogram encoding is unchanged.
- Degenerate inputs (unique points `< num_bins`, empty embeddings, single sample) breaking the `len(colors) == num_bins` invariant or k-means: define explicit behaviour (shrink `n_clusters` then pad, or documented alternative) and pin it with tests (Task 6/9).
- Out-of-gamut centroids raising in `LabColor.__post_init__`: clamp every centroid via `LabColor.clamp`/`np.clip` before construction; assert with the clamping test.
- Non-determinism across runs undermining reproducibility/A-B: fix `random_state=seed`, prefer `MiniBatchKMeans` for the 4,096-cluster scale, and assert palette equality under a fixed seed; resolve `n_init` per the Open Question (ROADMAP `'auto'` vs repo precedent `1`/`10`).
- Dependency on `002-p0-2-structure-preserving-training`: a palette fitted to noise is no better than uniform; gate the "honest VQ" claim on `009`'s A/B reconstruction comparison rather than asserting improvement in this step.
- Memory/time fitting 4,096 clusters: `MiniBatchKMeans` with `batch_size=min(256, num_samples)` (matching `pq_compression_baseline.py`) bounds memory; emit the fit-duration metric to catch regressions.
- New dependency risk: none — `scikit-learn` (`MiniBatchKMeans`) and `scipy` are already declared in `setup.cfg`/`pyproject.toml` and already used in `pq_compression_baseline.py`/`structured_pytorch_color_mapper.py`, so `pip-audit` needs no new entry and stays green.
- Extra constructor parameter on the use case breaking existing wiring/tests: make `codebook_factory` optional with uniform as the default mode, and update the CLI wiring and use-case tests in the same change set so coverage stays 100%.
