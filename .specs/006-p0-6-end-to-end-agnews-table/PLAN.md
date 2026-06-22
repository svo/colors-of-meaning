# Plan: Run end-to-end on AG News and fill the results table

## Implementation Strategy

This is a delivery step, not a code feature. The plan is to execute the existing
pipeline end-to-end on AG News, capture three artifacts (the README table row, a
committed run config, the structure-preservation correlation), and document a
reproducible command so the numbers regenerate within tolerance.

The work is gated on the full upstream P0 chain. The numbers are only trustworthy
once every one of these has landed, and each is cross-referenced by its canonical
spec directory:

- `001-p0-1-lab-emd-distance` (perceptual Lab EMD).
- `002-p0-2-structure-preserving-training` (real training objective).
- `003-p0-3-shuffle-stratify-sampling` (class-balanced AG News slice).
- `004-p0-4-eval-mapper-coverage-ef-fix` (`eval` loads every mapper; `ef >= k`).
- `005-p0-5-determinism-structure-metric` (reproducible seeding; the Spearman
  structure-preservation metric this step reports).

Sequencing within this step:

1. Confirm all five upstream specs are merged and `tox` is green on the integrated
   branch (do not run before the chain is complete — a partial bundle yields a
   meaningless number).
2. Choose and commit the run config (reuse/extend `configs/base.yaml`), pinning seed,
   device, `max_samples`, distance settings, and any field upstream steps added
   (e.g. `distance.sinkhorn_reg` from `001`).
3. Run `train -> encode -> eval` on AG News and capture Accuracy / Macro-F1 / MRR.
4. Run the structure-preservation evaluator from `005` and capture the Spearman
   correlation.
5. Edit `README.MD`: replace the `Color Method | TBD | TBD` row with the real
   numbers, record the correlation, and document the exact reproducible command and
   artifact paths.
6. Decide the artifact-storage and tolerance questions (see Risks), record the chosen
   tolerance in the README next to the command.
7. Run `tox` to confirm all eight gates pass and coverage stays 100%.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes.

### Application Layer (`src/colors_of_meaning/application/`)

No changes.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No changes.

### Interface Layer (`src/colors_of_meaning/interface/`)

No code changes. Existing CLI entry points (`train.py`, `encode.py`, `eval.py`) are
invoked unchanged via their `tox` environments.

### Shared Layer (`src/colors_of_meaning/shared/`)

No code changes. The committed run config is YAML consumed by the existing
`SynestheticConfig.from_yaml`; prefer extending `configs/base.yaml` over adding a new
file unless a distinct AG News run config is warranted.

## Dependency Injection

No changes. The CLI uses factory functions plus constructor injection; no Lagom
container is involved in the CLI path. This step adds no new components and no new
wiring.

## Task List

1. [ ] interface: verify `001`..`005` are merged and `tox` is green on the
   integrated branch before any run (gate on the full chain).
2. [ ] shared: select and commit the AG News run config (reuse/extend
   `configs/base.yaml`), pinning `training.seed`, `training.device: cpu`,
   `dataset.max_samples`, and any upstream-added field (e.g.
   `distance.sinkhorn_reg`).
3. [ ] interface: run `tox -e train -- --config <run-config>` to produce the
   projector and codebook artifacts.
4. [ ] interface: run `tox -e encode -- --config <run-config>` to persist the
   color-histogram artifact for the test split.
5. [ ] interface: run `tox -e eval -- --dataset ag_news --method color` and capture
   Accuracy, Macro-F1, and MRR.
6. [ ] infrastructure: run the structure-preservation evaluator from
   `005-p0-5-determinism-structure-metric` and capture the Spearman correlation.
7. [ ] interface: edit `README.MD` to replace the
   `| Color Method | TBD | TBD | Train model first |` row with the real Accuracy /
   Macro-F1 and a note linking the reproducible command and committed run config.
8. [ ] interface: document in `README.MD` the exact `train -> encode -> eval` command
   sequence, the artifact paths, the command that prints the correlation, and the
   stated regeneration tolerance.
9. [ ] interface: record the reported structure-preservation correlation in the
   README alongside the accuracy figures.
10. [ ] shared: resolve the artifact-storage decision (commit vs release asset vs
    local-only) and reflect the chosen paths in the README and `.gitignore` as needed.
11. [ ] interface: run `tox` and confirm all eight gates pass and coverage is 100%.

## Testing Strategy

The headline result is empirical output, not a unit-testable return value: a trained
model's accuracy and a measured correlation cannot be pinned to an exact constant in
a unit test, and re-running the pipeline inside `tox` would be far too slow and
non-deterministic across environments. This step is verified primarily by a
documented-command smoke check plus keeping `tox` and coverage green.

- Primary verification: run the documented `train -> encode -> eval` sequence on AG
  News once on the reference environment, observe that it completes and emits
  Accuracy / Macro-F1 / correlation, and confirm the README numbers match what was
  observed.
- Reproducibility check: with the committed run config and fixed seed, re-run the
  documented command and confirm the regenerated Accuracy, Macro-F1, and correlation
  fall within the stated tolerance (this exercises the determinism from `005`).
- Quality gates: `tox` must stay green and coverage must remain 100%. Because this
  step only edits documentation and YAML config, it adds no new executable code paths
  to cover; the existing CLI and pipeline tests from `001`..`005` continue to provide
  coverage.
- Optional small regression guard (welcome, not required): a lightweight check on the
  committed run config (for example, asserting it parses via `SynestheticConfig` and
  carries the pinned `seed`/`device`/`max_samples` and any upstream-added field used
  by the run) so a later config edit cannot silently invalidate the published numbers.
  Tests follow house rules: one logical assertion each, `test_should_..._when_...`
  naming, `assertpy` for config-entity assertions, no comments.

## Observability Plan

No new instrumentation. The run relies on existing CLI progress output and the
`_print_results` summary (Accuracy, Macro-F1, MRR, Recall@k, bits-per-token) in
`interface/cli/eval.py`. The durable record of this step is documentary: the README
table, the committed run config, and the reported correlation.

## Risks and Mitigations

- Risk: running before the upstream chain lands yields a meaningless number.
  Mitigation: Task 1 gates the run on all of `001`..`005` being merged and `tox`
  green.
- Risk: results drift across hardware or dependency versions, breaking "within
  tolerance". Mitigation: pin seed and `device: cpu` in the committed run config,
  state an explicit tolerance band in the README, and record the environment used.
- Risk: the canonical "Color Method" row is ambiguous when multiple mapper variants
  exist. Mitigation: decide up front (Open Question in SPEC) whether the row reports
  the unconstrained mapper or the best variant selected by the `005` structure
  metric, and reflect that single choice in the table.
- Risk: large trained artifacts bloat the repository if committed naively.
  Mitigation: resolve the storage decision in Task 10 (local-only reproducible output
  vs release asset) and keep heavy artifacts out of version control via `.gitignore`
  while committing only the lightweight run config.
- Risk: a later config edit silently invalidates the published numbers. Mitigation:
  the optional config-parse regression guard in the Testing Strategy.
- Risk: the README note and the committed config drift apart over time. Mitigation:
  have the README reference the committed run config path explicitly so the two are
  reviewed together.
