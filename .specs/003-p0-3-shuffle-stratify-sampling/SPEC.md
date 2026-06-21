# Feature: Shuffle/Stratify Dataset Sampling

## Overview

When a `max_samples` limit is configured, all three dataset adapters
(`ag_news`, `imdb`, `newsgroups`) currently take a **contiguous head slice** of
the underlying dataset with no shuffling. The HuggingFace IMDB dataset
(`stanfordnlp/imdb`) is stored ordered negative-then-positive on both `train`
and `test` splits, so the first N rows are **single-class**. With a typical
`max_samples` of 12,000, both the train fit and the test set contain exactly one
label, which makes every classifier report a meaningless 100% accuracy and macro
F1. This is the root cause that blocks an honest AG News results table (P0-6).

This feature replaces the head-slice with a **seeded shuffle**, and, where label
information is available before truncation, a **stratified** subsample that
preserves the per-class proportions of the full split. The configured
`TrainingConfig.seed` (default 42) is threaded from configuration through the
application and interface layers down to the dataset port so that sampling is
reproducible and deterministic across runs.

The minimal, verifiable outcome is: an IMDB `max_samples` slice contains both
the `negative` and the `positive` class, and repeated calls with the same seed
return the same samples in the same order.

## User Stories

- As a researcher running evaluations, I want a `max_samples`-limited IMDB slice
  to contain both classes so that reported accuracy reflects real classification
  performance rather than a degenerate single-class baseline.
- As a researcher comparing methods, I want sampling to honour the configured
  `seed` so that two runs with the same configuration draw the same subset and
  results are reproducible.
- As a maintainer, I want the per-class balance of a sampled split to track the
  full split (stratification where labels are known) so that subsampling does
  not silently distort class priors.
- As a maintainer, I want the seed to flow through configuration rather than
  being hard-coded in an adapter so that determinism is controlled in one place.

## Acceptance Criteria

- [ ] Given the IMDB adapter over a fixture whose rows are ordered all-negative
      then all-positive, when `get_samples("train", max_samples=N, seed=S)` is
      called with `N` smaller than the row count, then the returned samples
      contain at least one `label == 0` and at least one `label == 1`.
- [ ] Given the IMDB adapter over the same ordered fixture, when
      `get_samples("test", max_samples=N, seed=S)` is called, then the returned
      samples contain both classes.
- [ ] Given any dataset adapter, when `get_samples(split, max_samples=N,
      seed=S)` is called twice with identical `split`, `N`, and `S`, then both
      calls return the same texts and labels in the same order.
- [ ] Given any dataset adapter, when `get_samples(split, max_samples=N, seed=S1)`
      and `get_samples(split, max_samples=N, seed=S2)` are called with `S1 != S2`
      over a fixture large enough to permit distinct orderings, then the two
      results differ in order (shuffle actually depends on the seed).
- [ ] Given a fixture whose full split has a known class distribution, when
      `get_samples(split, max_samples=N, seed=S)` is called, then every class
      present in the full split is represented in the slice in proportion to its
      full-split frequency, with rounding such that the returned count equals
      `min(N, full_split_size)`.
- [ ] Given `max_samples` is `None`, when `get_samples(split, seed=S)` is called,
      then all rows of the split are returned (no truncation), shuffled
      deterministically by `seed`.
- [ ] Given the configured `TrainingConfig.seed`, when the `eval` CLI runs an
      evaluation, then the seed is passed through `EvaluateUseCase.execute` into
      `DatasetRepository.get_samples` for both the train and test splits.
- [ ] Given the configured `TrainingConfig.seed`, when the `train` CLI loads a
      labelled dataset for the supervised mapper, then the seed is passed into
      `DatasetRepository.get_samples`.
- [ ] Given the full quality gate (`tox`), when the feature is complete, then all
      8 gates pass with 100% coverage and no comments in production code.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

- `repository/dataset_repository.py`: extend the `get_samples` port signature to
  accept an optional `seed: Optional[int] = None` parameter:
  `get_samples(self, split: str, max_samples: Optional[int] = None, seed:
  Optional[int] = None) -> List[EvaluationSample]`. This is the **chosen
  approach** (see Dependency Injection and Open Questions). The abstract method
  body remains `raise NotImplementedError`.
- `model/evaluation_sample.py`: **No changes.** The existing frozen dataclass
  (`text`, `label`, `split`) already exposes `label` for stratification grouping
  and `text` for identity comparison in tests.

### Application Layer (`src/colors_of_meaning/application/`)

- `use_case/evaluate_use_case.py`: add an optional `seed: Optional[int] = None`
  parameter to `execute` and forward it into both
  `dataset_repository.get_samples(split="train", max_samples=..., seed=seed)` and
  the matching `split="test"` call. No business logic beyond delegation.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `dataset/imdb_dataset_adapter.py`: replace the `enumerate`/`break` head-slice
  with: load the full split, build `EvaluationSample` rows, then apply a shared
  seeded **stratified** subsample helper before returning. The IMDB labels are
  always known, so stratification applies.
- `dataset/ag_news_dataset_adapter.py`: same change; AG News labels are known, so
  stratify.
- `dataset/newsgroups_dataset_adapter.py`: same change; 20 Newsgroups labels are
  known, so stratify.
- Introduce one shared private sampling helper used by all three adapters so the
  shuffle/stratify/truncate logic lives in a single place (a new module under
  `infrastructure/dataset/`, e.g. `seeded_sampler.py`, kept inside the
  infrastructure layer). The helper uses `numpy`'s seeded
  `numpy.random.default_rng(seed)` for the permutation and per-class allocation;
  `numpy` is already a declared dependency. No new third-party dependency is
  required.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `cli/eval.py`: pass `config.training.seed` into
  `evaluate_use_case.execute(bits_per_token=..., max_samples=..., seed=...)`.
- `cli/train.py`: in `_load_supervised_data`, pass `config.training.seed` into
  `dataset_adapter.get_samples(split=..., max_samples=..., seed=...)`.
- `api/`: **No changes.** No API controller consumes `DatasetRepository`.

### Shared Layer

- `shared/synesthetic_config.py`: **No changes.** `TrainingConfig.seed` (default
  42) and `DatasetConfig.max_samples` already exist; this feature only threads
  the existing `seed` value, it does not add configuration.

## API Contracts

No API changes. No FastAPI controller depends on `DatasetRepository` or dataset
sampling, so no Pydantic DTO is added or altered. (No changes.)

## CLI Impact

- `eval` (`interface/cli/eval.py`): behaviour change only — the configured
  `seed` is now threaded into sampling; no new CLI flag is introduced (seed
  continues to come from `configs/*.yaml` via `TrainingConfig.seed`). Output and
  arguments (`EvalArgs`) are unchanged.
- `train` (`interface/cli/train.py`): behaviour change only for the supervised
  path — the configured `seed` is threaded into the labelled-dataset load. No new
  CLI flag; `TrainArgs` is unchanged.

## Dependency Injection

- **Chosen approach: thread the seed through the `get_samples` port signature.**
  The seed is a per-call sampling parameter that travels alongside the existing
  per-call `max_samples`; both describe how a particular request samples a split,
  not a fixed property of an adapter instance. Keeping them together preserves a
  single, coherent call contract and avoids stateful adapters. Wiring stays as
  today: the API uses Lagom `Container()` in `interface/api/main.py`; the CLI
  wires adapters via the existing factory functions (`_create_dataset_adapter`
  in `train.py`, `_setup_dataset` in `eval.py`) plus constructor injection. No
  new container registrations are required because the seed flows as a method
  argument resolved from `SynestheticConfig.training.seed`, not as an injected
  collaborator.
- The alternative (inject the seed via adapter construction) is recorded in Open
  Questions.

## Observability

- The dataset adapters are infrastructure I/O boundaries. A single structured
  log line per `get_samples` call recording `split`, `max_samples`, `seed`, the
  full-split size, and the returned size aids debugging of the single-class
  regression and confirms the seed actually reached the adapter. Logging follows
  the existing observability conventions (structured, correlation-id) used
  elsewhere in `infrastructure/observability/`. No new metrics are mandated by
  this step; emit at most one log line and keep it side-effect-free with respect
  to the returned samples.

## Open Questions

- **Alternative DI shape — seed via adapter construction.** Instead of widening
  the port, the seed could be a constructor argument on each adapter
  (`IMDBDatasetAdapter(seed=42)`), set once at wiring time in
  `_create_dataset_adapter` / `_setup_dataset` and in the Lagom container. This
  keeps the `get_samples` signature unchanged and the existing adapter unit tests
  that assert the exact `load_dataset`/`fetch_20newsgroups` call arguments stay
  valid without a seed argument. The cost is a stateful adapter and a second
  place (construction) that must agree with `max_samples` (a per-call argument),
  splitting the sampling contract across two call sites. We chose the port-
  signature approach for contract cohesion; revisit if a future caller needs the
  same adapter instance to sample with different seeds, or if widening the port
  is judged too invasive to the abstract interface.
- **Stratification rounding policy.** When `max_samples` does not divide evenly
  across classes, the per-class counts need a deterministic remainder
  distribution so the total equals `min(max_samples, full_split_size)` exactly.
  Proposed: floor each class's proportional share, then assign leftover slots to
  classes by descending fractional remainder (ties broken by ascending label).
  Confirm this is acceptable versus a simpler proportional-with-largest-remainder
  variant.
- **Rare-class guarantee.** Pure proportional stratification can floor a very
  rare class to zero in a small slice. Should the helper guarantee at least one
  sample per present class when `max_samples >= num_classes` (which also
  satisfies the IMDB both-classes acceptance test directly), accepting a small
  deviation from exact proportions? Proposed: yes, guarantee at least one per
  present class when the budget allows.
- **Train/test seed coupling.** `EvaluateUseCase.execute` passes one `seed` to
  both the train and test `get_samples` calls. Confirm a single shared seed for
  both splits is desired (it is reproducible and simplest); the splits are
  independent datasets so identical seeds do not cause leakage.
