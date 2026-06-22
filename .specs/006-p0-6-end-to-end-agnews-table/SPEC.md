# Feature: Run end-to-end on AG News and fill the results table

## Overview

This step produces the headline empirical result the blog post has never reported.
It runs the full color pipeline (`train -> encode -> eval`) on AG News, then records
the outcome in three places: the README "Current Performance (AG News Dataset)"
table, a committed run configuration, and a reported structure-preservation
correlation.

This is an integration and delivery step, not a new code feature. No new domain,
application, infrastructure, or interface code is required beyond what the upstream
P0 steps deliver. The substance is a documented, reproducible command sequence; a
filled README table with real Accuracy / Macro-F1 next to the existing TF-IDF
(90.63% / 90.61%) and HNSW (91.99% / 91.97%) baselines; the run config that
generated those numbers; and the Spearman structure-preservation correlation from
`005-p0-5-determinism-structure-metric`.

The reported numbers are only trustworthy once the entire upstream P0 chain has
landed. Each upstream step removes one independent reason the color-method number
would be invalid; until all five are merged, any value written into the table would
be meaningless. This step therefore depends on, and must be executed after:

- `001-p0-1-lab-emd-distance` — a true perceptual Lab Earth-Mover distance, so
  retrieval reflects color geometry rather than bin-index ordering.
- `002-p0-2-structure-preserving-training` — a real training objective, so similar
  meanings map to similar colors rather than to uniform-random Lab targets.
- `003-p0-3-shuffle-stratify-sampling` — seeded shuffle/stratify before truncation,
  so the AG News split is class-balanced rather than a contiguous head slice.
- `004-p0-4-eval-mapper-coverage-ef-fix` — `eval` can load every mapper variant and
  uses `ef >= num_candidates`, so retrieval is correct and the chosen checkpoint
  actually loads.
- `005-p0-5-determinism-structure-metric` — honoured seeding (reproducible Lab
  outputs) and the Spearman structure-preservation evaluator whose value this step
  reports.

## User Stories

- As a researcher, I want a single documented command sequence that runs
  `train -> encode -> eval` on AG News so that I can regenerate the reported
  color-method numbers without reverse-engineering CLI arguments.
- As a reader of the README, I want the "Color Method" row to show real Accuracy and
  Macro-F1 next to the TF-IDF and HNSW baselines so that I can judge the thesis
  against a concrete result instead of a `TBD`.
- As a reviewer, I want the exact run configuration committed alongside the result so
  that the published numbers are attributable to a specific, version-controlled
  configuration.
- As a researcher, I want the structure-preservation correlation reported next to the
  accuracy so that the blog's central claim (similar meanings map to similar colors)
  is measured directly, not merely inferred from classification accuracy.

## Acceptance Criteria

- [ ] Given all of `001`..`005` are merged, when the documented command sequence is
  run on AG News, then it produces a color-method Accuracy and Macro-F1 and a
  structure-preservation Spearman correlation without manual intervention.
- [ ] Given the run has completed, when a reader opens `README.MD`, then the row
  `| Color Method | TBD | TBD | Train model first |` is replaced by the real
  Accuracy / Macro-F1 with a note that points to the reproducible command and the
  committed run config.
- [ ] Given the run config is committed, when a reviewer inspects the repository,
  then the exact configuration that generated the reported numbers is present under
  version control and referenced from the README.
- [ ] Given the structure-preservation correlation is produced by
  `005-p0-5-determinism-structure-metric`, when the README result is read, then the
  reported correlation is recorded alongside the accuracy figures.
- [ ] Given the configured seed and the committed run config, when the documented
  command is re-run on the same environment, then the regenerated Accuracy, Macro-F1,
  and correlation match the reported values within a stated tolerance.
- [ ] Given this delivery step, when `tox` is run, then all eight gates pass and
  coverage remains 100% (the documentation and config edits introduce no uncovered
  code paths).

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. The structure-preservation port and any distance ports used by the run
are delivered by the upstream P0 steps (`001`, `005`).

### Application Layer (`src/colors_of_meaning/application/`)

No changes. The orchestration in `TrainColorMappingUseCase`, `EncodeDocumentUseCase`,
and `EvaluateUseCase` is reused as-is; this step invokes existing use cases through
the CLI rather than adding new ones.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No changes. The distance calculator, mappers, dataset adapters, classifiers, metrics
calculator, and the structure-preservation evaluator are all delivered upstream. This
step only executes them.

### Interface Layer (`src/colors_of_meaning/interface/`)

No code changes. The existing CLI entry points are used unchanged:

- `interface/cli/train.py` (`TrainArgs`: `--config`, `--dataset-path`,
  `--output-model`, `--output-codebook`, `--mapper-type`).
- `interface/cli/encode.py` (`EncodeArgs`: `--config`, `--split`, `--dataset-path`,
  `--model-path`, `--codebook-name`, `--output-path`).
- `interface/cli/eval.py` (`EvalArgs`: `--config`, `--dataset`, `--method`,
  `--model-path`, `--codebook-path`, `--k-neighbors`; the `--mapper-type` flag and
  matching-network construction are added by `004-p0-4-eval-mapper-coverage-ef-fix`).

### Shared Layer

No code changes. The run config reuses the existing `synesthetic_config` schema
loaded by `SynestheticConfig.from_yaml`. The committed run config is a YAML file in
the existing `configs/` family (reuse/extend `configs/base.yaml`); any field added by
upstream steps (e.g. `distance.sinkhorn_reg` from `001`) is set there if the run
exercises it.

## API Contracts

No changes. This step touches no HTTP endpoints; the API surface is untouched.

## CLI Impact

No new CLI commands or flags are introduced by this step. The deliverable is a
documented sequence over the existing `tox` environments, all of which pass posargs
through to the CLI entry points:

1. `tox -e train -- --config configs/base.yaml`
   (trains the projector and writes the codebook; `--mapper-type` selects the
   variant once `002`/`004` make non-default mappers worth evaluating).
2. `tox -e encode -- --config configs/base.yaml`
   (produces the color-histogram artifact for the test split, used for inspection
   and palette queries).
3. `tox -e eval -- --dataset ag_news --method color`
   (re-encodes internally and reports Accuracy / Macro-F1 / MRR).

The README must document this exact sequence, the artifact paths used, and the
command that prints the structure-preservation correlation from `005`. Note that the
color `eval` path re-encodes documents internally, so step 2 is for the persisted
artifact (queries, inspection) rather than a precondition of step 3.

## Dependency Injection

No changes. The CLI wires dependencies through factory functions and constructor
injection (e.g. `_create_color_mapper`, `_create_classifier`,
`_create_color_classifier`); there is no Lagom container in the CLI path. This step
adds no new wiring and introduces no new components to inject.

## Observability

No new instrumentation. The run relies on the existing CLI progress output and the
`_print_results` summary in `interface/cli/eval.py` (Accuracy, Macro-F1, MRR,
Recall@k, bits-per-token). The recorded artifacts of this step are documentary: the
README table, the committed run config, and the reported correlation value.

## Open Questions

- Where should trained artifacts (`artifacts/models/projector.pth`, the codebook
  pickle, the encoded test documents) live, and should any be committed, published as
  a release asset, or left as reproducible local output? The README currently implies
  local paths under `artifacts/`.
- What is the tolerance for "regenerates the reported numbers within tolerance"?
  This must be stated as an explicit band (for example, an absolute tolerance on
  Accuracy / Macro-F1 and on the Spearman correlation) and pinned to a fixed seed,
  device (`cpu` per the config), and dependency set, since results may drift across
  hardware or library versions.
- Which mapper variant's result is the canonical "Color Method" row — the
  unconstrained mapper, or the best-performing variant by structure-preservation
  checkpointing from `005`? If more than one is reported, the table layout needs to
  accommodate that.
