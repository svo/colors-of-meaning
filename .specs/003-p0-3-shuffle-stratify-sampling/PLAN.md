# Plan: shuffle-stratify-sampling

## Implementation Strategy

The defect is that `max_samples` truncation is a contiguous head slice, which on
the neg-then-pos-ordered IMDB dataset yields single-class train and test sets and
a meaningless 100% score. The fix is one shared, seeded, stratified subsampling
helper applied by all three dataset adapters after they have materialised the
full split into `EvaluationSample` rows, plus threading the existing
`TrainingConfig.seed` from configuration through `EvaluateUseCase.execute` and
the CLI down to the widened `DatasetRepository.get_samples` port.

Work proceeds test-first (TDD) in dependency order: domain port, application use
case, infrastructure helper and adapters, then interface wiring, with tests
written before each implementation. All sampling tests use synthetic in-memory
fixtures — `load_dataset` and `fetch_20newsgroups` are patched exactly as in the
existing adapter tests, so no HuggingFace or sklearn download occurs. The
acceptance test (both IMDB classes present in a slice) is realised over a patched
fixture ordered all-negative-then-all-positive.

Determinism is provided by `numpy.random.default_rng(seed)` (numpy is already a
declared dependency); no new third-party package is introduced, so no setup.cfg /
pyproject / pip-audit change is needed for this step. The existing adapter unit
tests that assert the exact `load_dataset` / `fetch_20newsgroups` call arguments
remain valid because the underlying-loader call is unchanged; tests that assert
returned counts are updated to pass a `seed` and to reflect stratified counts.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `repository/dataset_repository.py`: widen the abstract `get_samples` to
  `get_samples(self, split: str, max_samples: Optional[int] = None, seed:
  Optional[int] = None) -> List[EvaluationSample]`. Body stays
  `raise NotImplementedError`. No other domain change. `model/evaluation_sample.py`
  is unchanged (already exposes `label` and `text`).

### Application Layer (`src/colors_of_meaning/application/`)

- `use_case/evaluate_use_case.py`: add `seed: Optional[int] = None` to `execute`
  and forward it into both the `split="train"` and `split="test"` calls to
  `self.dataset_repository.get_samples`. No other logic changes.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- New module `dataset/seeded_sampler.py`: a single pure function that takes a
  `List[EvaluationSample]`, `max_samples`, and `seed`, and returns a
  deterministically shuffled, label-stratified subsample. It groups by
  `sample.label`, allocates per-class counts proportionally (floor + largest
  fractional remainder, ties by ascending label; at least one per present class
  when the budget allows), draws per-class indices and a final ordering using
  `numpy.random.default_rng(seed)`, and returns exactly
  `min(max_samples, len(samples))` rows (all rows shuffled when `max_samples` is
  `None`).
- `dataset/imdb_dataset_adapter.py`, `dataset/ag_news_dataset_adapter.py`,
  `dataset/newsgroups_dataset_adapter.py`: remove the `enumerate`/`break`
  head-slice; build the full list of `EvaluationSample`, then return
  `seeded_subsample(samples, max_samples, seed)`. Accept and pass through the new
  `seed` parameter on `get_samples`.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `cli/eval.py`: pass `config.training.seed` into `evaluate_use_case.execute(...)`.
- `cli/train.py`: in `_load_supervised_data`, pass `config.training.seed` into
  `dataset_adapter.get_samples(...)`.
- No API controller change.

### Shared Layer (`src/colors_of_meaning/shared/`)

- No changes. `TrainingConfig.seed` and `DatasetConfig.max_samples` already exist.

## Dependency Injection

Seed is threaded as a per-call method argument (matching `max_samples`), not as
an injected collaborator, so no Lagom container registration changes are needed.
Adapters continue to be constructed by the existing CLI factories
(`_create_dataset_adapter`, `_setup_dataset`) and the API `Container()`; the seed
value is read from `SynestheticConfig.training.seed` at the call site and passed
down. The alternative (constructor-injected seed) is recorded in SPEC Open
Questions.

## Task List

1. [ ] domain: add a failing port-contract test that a stub `DatasetRepository`
   implementing the widened `get_samples(split, max_samples, seed)` signature is
   instantiable and callable with a `seed` keyword.
2. [ ] domain: widen `get_samples` in `repository/dataset_repository.py` to
   accept `seed: Optional[int] = None`.
3. [ ] application: add a failing test that `EvaluateUseCase.execute(seed=S)`
   forwards `seed=S` into `get_samples` for the `train` split (mock repository,
   assert call kwargs).
4. [ ] application: add a failing test that `EvaluateUseCase.execute(seed=S)`
   forwards `seed=S` into `get_samples` for the `test` split.
5. [ ] application: add `seed` to `EvaluateUseCase.execute` and forward it to
   both `get_samples` calls.
6. [ ] infrastructure: add failing tests for `seeded_sampler.seeded_subsample` —
   stratified slice contains every class present in the input; returned count
   equals `min(max_samples, len(samples))`; same seed reproduces identical
   ordering; different seeds produce different ordering; `max_samples=None`
   returns all rows.
7. [ ] infrastructure: implement `dataset/seeded_sampler.py` with
   `numpy.random.default_rng`-based shuffle + proportional stratification +
   rare-class floor, passing task 6.
8. [ ] infrastructure: add the acceptance test — patch
   `imdb_dataset_adapter.load_dataset` with an all-negative-then-all-positive
   fixture; assert `get_samples("train", max_samples=N, seed=S)` contains both
   `label == 0` and `label == 1`.
9. [ ] infrastructure: add the matching acceptance test for the `test` split of
   IMDB (both classes present).
10. [ ] infrastructure: refactor `imdb_dataset_adapter.py` to build the full
    sample list and return `seeded_subsample(samples, max_samples, seed)`;
    accept `seed` on `get_samples`.
11. [ ] infrastructure: refactor `ag_news_dataset_adapter.py` the same way
    (stratified), accepting `seed`.
12. [ ] infrastructure: refactor `newsgroups_dataset_adapter.py` the same way
    (stratified), accepting `seed`.
13. [ ] infrastructure: update the existing per-adapter count/limit tests to pass
    a `seed` and to assert stratified counts; keep the existing
    `load_dataset` / `fetch_20newsgroups` call-argument assertions intact.
14. [ ] interface: update `cli/eval.py` to pass `config.training.seed` into
    `evaluate_use_case.execute`, and add/adjust its test to assert the seed is
    forwarded.
15. [ ] interface: update `cli/train.py` `_load_supervised_data` to pass
    `config.training.seed` into `get_samples`, and add/adjust its test to assert
    the seed is forwarded.
16. [ ] tests: add a `pytest-archon` assertion (or confirm the existing rule in
    `tests/colors_of_meaning/test_synesthetic_architecture.py`) that
    `infrastructure.dataset.*` does not import `interface` or `application`,
    keeping the new `seeded_sampler` module within infrastructure.
17. [ ] tests: add structured-logging coverage for the one log line emitted per
    `get_samples` call (if logging is added), asserting it does not mutate the
    returned samples.
18. [ ] all: run `tox` and resolve every gate (flake8, black, bandit, semgrep,
    pip-audit, radon, xenon, mypy) to green at 100% coverage.

## Testing Strategy

- **Framework split:** ML/numerical and adapter behaviour (sampling counts,
  class presence, determinism) use plain `assert` and `pytest.raises`; any
  base-entity-style assertions use `assertpy`'s `assert_that`. Sampling
  acceptance tests are numerical/domain, so plain `assert` is appropriate.
- **One logical assertion per test;** related numerical checks on the same
  sampling result (e.g. both `label == 0` and `label == 1` present) may be
  grouped as one logical assertion about class coverage. Each distinct property
  (count equals `min(...)`, same-seed reproducibility, different-seed variation,
  `None` returns all) is its own test.
- **Naming:** every test follows `test_should_<behaviour>_when_<condition>`, e.g.
  `test_should_include_both_classes_when_imdb_slice_is_sampled_with_seed`,
  `test_should_return_identical_order_when_same_seed_is_reused`,
  `test_should_return_all_rows_when_max_samples_is_none`.
- **No network / no download:** `load_dataset` and `fetch_20newsgroups` are
  patched with synthetic fixtures exactly as in the current adapter tests; the
  IMDB acceptance fixture is an explicit all-negative-then-all-positive list. No
  real HuggingFace or sklearn fetch is performed in any unit test.
- **Architecture:** a `pytest-archon` `archrule` (in
  `tests/colors_of_meaning/test_synesthetic_architecture.py`) confirms the new
  `seeded_sampler` module stays within `infrastructure` and does not import
  `application` or `interface`; the existing domain-independence rule continues to
  guard the widened port.
- **Coverage and gates:** verified only via `tox` (all 8 gates), never `pytest`
  alone; target is 100% coverage including the rare-class branch, the
  `max_samples=None` branch, and the remainder-distribution branch of the helper.

## Observability Plan

Optionally emit one structured log line per `get_samples` call recording `split`,
`max_samples`, `seed`, full-split size, and returned size, using the existing
`infrastructure/observability/` logger conventions (structured, correlation-id).
The log line must be side-effect-free with respect to the returned samples and is
covered by a dedicated test. No new metrics are introduced for this step.

## Risks and Mitigations

- **Risk: existing adapter tests break.** The current tests assert exact loader
  call arguments and small returned counts. *Mitigation:* the loader call is
  unchanged so those assertions stay; only count/limit tests are updated to pass
  a `seed` and reflect stratified counts (task 13).
- **Risk: widening the port is an interface change that ripples to all
  implementers.** *Mitigation:* `seed` is optional with a `None` default, so the
  three production adapters and any test stubs remain backward-compatible;
  callers opt in by passing the seed.
- **Risk: stratification rounding produces the wrong total or drops a rare
  class.** *Mitigation:* deterministic largest-fractional-remainder allocation
  with a one-per-present-class floor when budget allows, each branch covered by a
  dedicated test (tasks 6-7); the IMDB both-classes acceptance test (tasks 8-9)
  guards the headline regression.
- **Risk: non-determinism from a global RNG.** *Mitigation:* use a local
  `numpy.random.default_rng(seed)` instance, never the global `numpy.random`
  state, and assert same-seed reproducibility plus different-seed variation.
- **Risk: seed not actually reaching the adapter through the layers.**
  *Mitigation:* application-layer forwarding tests (tasks 3-4) and CLI forwarding
  tests (tasks 14-15) assert the seed is passed at every hop; optional log line
  records the received seed.
