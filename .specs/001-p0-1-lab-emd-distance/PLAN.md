# Plan: True Lab Earth-Mover Distance (POT)

## Implementation Strategy
Rewrite the existing `WassersteinDistanceCalculator` adapter in place (PREFER EDITING over creating) so it computes a true Wasserstein-1 / Earth-Mover distance over a precomputed perceptual cost matrix, using POT (`ot`). The `DistanceCalculator` port signature is unchanged, so the change is confined to one infrastructure file plus its construction sites. Key decisions:

- **Inject the codebook, precompute once.** The constructor receives a `ColorCodebook`, derives `coords = np.array([[c.l, c.a, c.b] for c in codebook.colors], dtype=np.float32)`, and builds `M = ot.dist(coords, coords, metric='euclidean')` a single time. `metric='euclidean'` is mandatory — `ot.dist`'s default is `sqeuclidean`, which would give squared-W2, not W1. The ~67 MB matrix is held on the instance and reused for every `compute_distance` call.
- **Exact by default, Sinkhorn when configured.** `compute_distance` returns `float(ot.emd2(doc1.histogram, doc2.histogram, M))`. When the new `distance.sinkhorn_reg` is a positive float, it instead returns `float(ot.sinkhorn2(doc1.histogram, doc2.histogram, M, sinkhorn_reg, method='sinkhorn_log'))` (`sinkhorn_log` is POT's numerically stable variant). Branch selection is decided once in the constructor (store a strategy/callable) to keep `compute_distance` low-complexity for `xenon`/`radon`.
- **Stricter validation.** Replace the bare `doc1.num_bins != doc2.num_bins` check with a same-codebook check: both documents' `num_bins` must equal the codebook's `num_bins`; otherwise raise `ValueError`.
- **Config field.** Add `sinkhorn_reg: Optional[float] = None` to `DistanceConfig`; YAML round-trips it automatically.
- **No new files in src.** Edit the calculator, the config dataclass, and the four CLI factories. No new test module is needed — extend the existing `test_wasserstein_distance_calculator.py`. New dependency `ot` goes in `setup.cfg`.
- **`smoothing_epsilon` stays distinct.** Do not touch Jensen-Shannon; document in code structure (via naming) that `smoothing_epsilon` and `sinkhorn_reg` are different fields with different consumers.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
- No edits. `domain/service/distance_calculator.py` port is satisfied unchanged. `domain/model/color_codebook.py` and `domain/model/colored_document.py` are read-only inputs. Confirm via the architecture test that `domain/*` still imports no `ot`/`scipy`.

### Application Layer (`src/colors_of_meaning/application/`)
- No edits. `compare_documents_use_case.py` and `query_by_palette_use_case.py` keep receiving the port by constructor injection; no infrastructure/interface imports introduced.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- Rewrite `infrastructure/ml/wasserstein_distance_calculator.py`:
  - `__init__(self, codebook: ColorCodebook, sinkhorn_reg: Optional[float] = None)`.
  - Build `coords` and `M = ot.dist(coords, coords, metric='euclidean')` once.
  - Select exact vs Sinkhorn strategy once based on `sinkhorn_reg`.
  - `compute_distance` validates both `num_bins` against the codebook, then returns the chosen OT value as `float`.
  - `metric_name` returns `"wasserstein"` (unchanged).
  - Remove `from scipy.stats import wasserstein_distance`; add `import ot`.
  - No comments; expressive private names (e.g. `_perceptual_cost_matrix`, `_codebook_coordinates`, `_compute_exact_emd`, `_compute_entropic_emd`).
- `infrastructure/evaluation/color_histogram_classifier.py` — no edit; it is the retrieval consumer of the port and inherits the new behaviour.

### Interface Layer (`src/colors_of_meaning/interface/`)
- `interface/cli/eval.py` `_create_color_classifier`: pass the already-loaded `codebook` and `config.distance.sinkhorn_reg` into `WassersteinDistanceCalculator(...)` (the test `test_eval.py` patches the symbol, so it stays mock-compatible).
- `interface/cli/query.py` `main`: reuse the already-loaded `codebook` in the `WassersteinDistanceCalculator(...)` construction; thread `sinkhorn_reg` if a config is loaded (query.py currently loads no `SynestheticConfig` — either load it or default `sinkhorn_reg=None`; prefer `None` to avoid widening the CLI surface — see Open Questions in SPEC).
- `interface/cli/compare.py` `main`: load the codebook (new — uses `FileColorCodebookRepository().load(...)`) and pass it plus `config.distance.sinkhorn_reg`; keep the `metric == "wasserstein"` branch.
- `interface/cli/visualize.py` `_create_classifier`: reuse the already-loaded `codebook` in the color branch.
- Pydantic DTOs / FastAPI controllers: no changes.

### Shared Layer (`src/colors_of_meaning/shared/`)
- `shared/synesthetic_config.py`: add `sinkhorn_reg: Optional[float] = None` to `DistanceConfig`. No change to `from_yaml`/`to_yaml` logic (dataclass `**`/`__dict__` handles it).
- `shared/lab_utils.py`: no edit.
- `configs/*.yaml`: optionally document `sinkhorn_reg` under `distance:`; safe to omit.

## Dependency Injection
- CLI factories construct `WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)` and inject it into `ColorHistogramClassifier` / `CompareDocumentsUseCase` (constructor injection, as today).
- Abstract → concrete: `DistanceCalculator` → `WassersteinDistanceCalculator(codebook, sinkhorn_reg)`.
- API `interface/api/main.py` Lagom container: no `DistanceCalculator` registration added in this step (no API consumer yet). Left as a documented hook for 011-p2-1-mount-query-retire-coconut.
- Test wiring: unit tests construct the calculator directly with a small synthetic `ColorCodebook`; CLI tests continue to patch `...cli.<module>.WassersteinDistanceCalculator` and assert it is called with the codebook (and `sinkhorn_reg`) argument.

## Task List
Ordered, each independently committable:
1. [ ] shared: add `sinkhorn_reg: Optional[float] = None` to `DistanceConfig` in `shared/synesthetic_config.py`; add a test asserting the default is `None` and that `from_yaml` reads a provided value. (Covers AC: config-gated Sinkhorn selection — config half.)
2. [ ] deps: add `ot` to `setup.cfg` `install_requires`; run `pip-audit` (via `tox`) and pin a minimum version if flagged. (Covers AC: new dep + pip-audit green.)
3. [ ] infrastructure: rewrite `infrastructure/ml/wasserstein_distance_calculator.py` to inject `ColorCodebook`, precompute `M = ot.dist(coords, coords, metric='euclidean')`, return `ot.emd2` (exact) or `ot.sinkhorn2(..., method='sinkhorn_log')` (when `sinkhorn_reg` set), and validate both documents against the codebook size. (Covers AC: perceptual ordering, identical→~0, precompute-once, same-codebook validation, exact-vs-Sinkhorn, metric_name.)
4. [ ] interface: rewire the four CLI factories (`eval.py`, `compare.py` incl. new codebook load, `query.py`, `visualize.py`) to inject the codebook and `sinkhorn_reg`. (Covers AC: CLI commands still run.)
5. [ ] tests: rewrite/extend `tests/colors_of_meaning/infrastructure/ml/test_wasserstein_distance_calculator.py` for the new constructor and perceptual properties; update the four CLI tests to expect the codebook argument; confirm architecture/boundary tests still pass. (Covers AC: perceptual close-small / distant-large, identical→0, validation error, observability log, 100% coverage, tox green.)

## Testing Strategy
- **One assertion per test**, ML/numerical tests may group related asserts on the same result; plain `assert`/`pytest.raises` for these ML calculator tests (matching the existing `test_wasserstein_distance_calculator.py` and `test_jensen_shannon_distance_calculator.py` style). Base/config-dataclass tests use `assertpy` (`assert_that`).
- **Names** follow `test_should_<behaviour>_when_<condition>`.
- **Core perceptual test (the headline acceptance):** build a tiny `ColorCodebook` of, say, three Lab colors where two are perceptually close (small ΔE) and one is far; create two histogram pairs — one moving unit mass between the close colors, one between the far colors — and assert `close_distance < far_distance` (and that the far distance is meaningfully larger). This is the property the old index implementation fails.
  - `test_should_yield_small_distance_when_mass_moves_between_perceptually_close_colors`
  - `test_should_yield_larger_distance_when_mass_moves_between_perceptually_distant_colors`
- Supporting ML tests: `test_should_return_zero_when_histograms_are_identical`; `test_should_raise_value_error_when_document_bins_do_not_match_codebook`; `test_should_use_sinkhorn_when_sinkhorn_reg_is_set` (assert it runs and returns a finite non-negative float, optionally that it approximates `emd2`); `test_should_use_exact_emd_when_sinkhorn_reg_is_none`; `test_should_return_wasserstein_when_metric_name_is_requested`.
- Config tests: `test_should_default_sinkhorn_reg_to_none_when_not_provided`; `test_should_read_sinkhorn_reg_when_present_in_yaml`.
- CLI tests: extend each patched-symbol test to assert `WassersteinDistanceCalculator` is constructed with the codebook (and `sinkhorn_reg`) so the wiring is pinned.
- **Boundary tests:** `pytest-archon` rules in `tests/colors_of_meaning/test_synesthetic_architecture.py` already assert `domain/*` imports nothing from infrastructure/interface/application; they must stay green (the `ot` import lives only in infrastructure).
- **CDCT:** none required — no service integration crosses a process boundary in this step.
- **Verification:** `tox` for the full 8 gates (never `pytest` alone); `tox -- tests/colors_of_meaning/infrastructure/ml/test_wasserstein_distance_calculator.py` for fast TDD loops; `tox -e format` before completion. 100% coverage required, including both `sinkhorn_reg`-set and `None` branches and the validation-error branch.

## Observability Plan
- Emit one structured log entry (with `correlation-id`) at calculator construction recording `num_bins`, cost-matrix shape, and the active strategy (`exact` vs `sinkhorn`); do **not** log inside `compute_distance` (hot reranking loop). Test asserts the log call happens once on construction.
- No new counters/gauges/histograms or tracing spans for the inner loop in this step; the unchanged `CompareDocumentsUseCase` keeps any existing use-case tracing.

## Risks and Mitigations
- **Risk:** 4096×4096 cost matrix (~67 MB `float32`) plus per-pair exact LP solving makes `eval` over a large corpus slow. → **Mitigation:** precompute `M` once (already core to the design); offer the Sinkhorn path via `sinkhorn_reg`; document a recommended reg for corpus-scale runs (decision deferred to 006-p0-6-end-to-end-agnews-table; tracked in SPEC Open Questions).
- **Risk:** forgetting `metric='euclidean'` yields squared distances (W2²), silently changing semantics and breaking threshold-based tests. → **Mitigation:** the perceptual close-vs-far test pins ordering; an explicit assertion that an identical pair returns ~0 and the metric name is unchanged; code review checklist item.
- **Risk:** `ot.emd2`/`ot.sinkhorn2` may require `float64` histograms or contiguous arrays; `ColoredDocument.histogram` is `float64` but `coords`/`M` are `float32`, and POT may warn or upcast. → **Mitigation:** decide `M` dtype in task 3 (SPEC Open Question), add a test that a realistic 4096-bin pair computes without error/warning.
- **Risk:** new `ot` dependency carries a pip-audit advisory or heavy transitive deps. → **Mitigation:** run `tox` (pip-audit gate) in task 2 before wiring; pin a clean minimum version if needed.
- **Risk:** `compute_distance` branching (exact vs Sinkhorn) pushes cyclomatic complexity past `xenon`/`radon` limits. → **Mitigation:** resolve the strategy once in `__init__` into a stored callable so `compute_distance` stays a straight-line method.
- **Risk:** CLI test mocks break because the calculator is now called with arguments. → **Mitigation:** update the four CLI tests in task 5 to assert the new call signature; they already patch the imported symbol, so no structural change is needed.
- **Risk:** `query.py` loads no `SynestheticConfig`, so `sinkhorn_reg` has no source there. → **Mitigation:** default to `None` (exact EMD) at that call site rather than widening the CLI; revisit only if corpus-scale palette query needs Sinkhorn.
