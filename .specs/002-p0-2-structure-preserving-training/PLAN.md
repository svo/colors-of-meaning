# Plan: structure-preserving-training

## Implementation Strategy
Replace the random-target regression in both learnable mappers with a structure-preserving objective, working strictly inside the infrastructure layer so the `ColorMapper` port, the application use case, and the CLI wiring stay untouched.

1. Unconstrained mapper (`infrastructure/ml/pytorch_color_mapper.py`): delete `_generate_targets`; in `train`, for each batch build a frozen-teacher target `gold = util.cos_sim(batch_embeddings, batch_embeddings).detach()`, build a student similarity from the 3-D Lab outputs, mask to off-diagonal entries, and minimise `MSE(student_offdiag, gold_offdiag)`. Extract small private helpers (`_teacher_similarity`, `_student_similarity`, `_offdiagonal`) to keep cyclomatic complexity within `xenon`/`radon` limits and avoid comments.
2. Supervised mapper (`infrastructure/ml/supervised_pytorch_color_mapper.py`): delete `_generate_targets`; remove `projection_targets` from the `train`/`_run_training_loop`/`_train_epoch`/`_train_batch`/`_compute_combined_loss` call chain; replace the dominant Lab MSE with a class-keyed contrastive/triplet term on `lab_output` using `self._training_labels`, and combine it with cross-entropy via a scale-normalised sum (L/100, a/b/127.5) or learnable uncertainty weights so the classification term is not swamped.
3. Tests: assert the new mechanism (near-duplicate co-location, teacher-ordering monotonicity, supervised term swap, scale normalisation, off-diagonal masking, determinism) and update the existing tests that referenced `_generate_targets`.
4. Cross-reference `005-p0-5-determinism-structure-metric` for the held-out similarity-discrepancy metric and seeded determinism; this step does not implement that metric but is validated by it.

Develop test-first (TDD): add the failing near-duplicate co-location test against the current random-target mapper, confirm it fails, then implement the objective.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
No changes. `domain/service/color_mapper.py` (`ColorMapper` ABC and `QuantizedColorMapper`) keeps its current signatures. Domain remains torch-free and pure. A `pytest-archon` boundary test continues to forbid torch/sklearn imports in `domain/`.

### Application Layer (`src/colors_of_meaning/application/`)
No changes. `application/use_case/train_color_mapping_use_case.py` already calls `self.color_mapper.train(embeddings=..., epochs=..., learning_rate=...)`; the behaviour change is fully behind the port. Determinism/checkpointing bootstrap edits belong to `005-p0-5-determinism-structure-metric`.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- `infrastructure/ml/pytorch_color_mapper.py`
  - Delete `_generate_targets` (lines 124-132).
  - Rewrite `train` (lines 79-114): drop the `targets = self._generate_targets(...)` precompute; per batch compute teacher cosine similarity (`util.cos_sim(...).detach()`), student Lab similarity from `self.network(batch_embeddings)`, and `nn.MSELoss()` over off-diagonal entries; keep Adam, batching, randperm shuffling, and the every-10-epoch print.
  - Add private helpers `_teacher_similarity`, `_student_similarity`, `_offdiagonal` to keep each method simple.
  - Import `from sentence_transformers import util` (already a declared dependency).
- `infrastructure/ml/supervised_pytorch_color_mapper.py`
  - Delete `_generate_targets` (lines 200-208).
  - Remove `projection_targets`/`projection_loss` plumbing from `train` (line 77), `_run_training_loop` (99-121), `_train_epoch` (123-146), `_train_batch` (148-167), and `_compute_combined_loss` (169-180).
  - Implement a class-keyed contrastive (or triplet) loss on `lab_output` keyed by `self._training_labels`, and combine with `cross_entropy` via scale-normalised Lab terms (L/100 into [0,1], a/b/127.5 into [-1,1]) or learnable uncertainty weights; retain `classification_weight`, AdamW, cosine LR schedule, gradient clipping, and best-state checkpointing.
- `infrastructure/ml/structured_pytorch_color_mapper.py`: no functional change in this step.

### Interface Layer (`src/colors_of_meaning/interface/`)
No changes. `interface/cli/train.py` `_create_color_mapper` and `set_training_labels` flow already supply everything the new objectives need. No new flag; `eval` mapper-reach is `004-p0-4-eval-mapper-coverage-ef-fix`.

### Shared Layer (`src/colors_of_meaning/shared/`)
No changes required. `shared/lab_utils.delta_e` (numpy CIE76) remains the reference perceptual distance; any batched Delta E used inside the student similarity is reimplemented in torch within the infrastructure adapter to keep `shared` torch-free. New structure-loss hyperparameters in `shared/synesthetic_config.py` are deferred pending the Open Questions; if added, `TrainingConfig`/`SupervisedMapperConfig` plus `from_yaml`/`to_yaml` gain round-trip tests.

## Dependency Injection
No new ports or Lagom registrations. CLI factory functions (`_create_color_mapper`) keep constructing the mappers and injecting them into `TrainColorMappingUseCase`; the API Lagom `Container()` in `interface/api/main.py` is untouched (training is not exposed via API). The change is internal to existing `ColorMapper` implementations resolved behind the existing port.

## Task List
1. [ ] domain: confirm `domain/service/color_mapper.py` needs no signature change; record the no-change decision (port stays `train(embeddings, epochs, learning_rate)`).
2. [ ] application: confirm `application/use_case/train_color_mapping_use_case.py` needs no change (behaviour is behind the port).
3. [ ] infrastructure: in `pytorch_color_mapper.py`, write the failing near-duplicate co-location test target behaviour, then delete `_generate_targets` and add `_teacher_similarity` (`util.cos_sim(...).detach()`).
4. [ ] infrastructure: add `_student_similarity` (cosine of mean-centred Lab, with a Delta E variant) and `_offdiagonal` masking helper in `pytorch_color_mapper.py`.
5. [ ] infrastructure: rewrite `pytorch_color_mapper.train` to optimise `MSE(student_offdiag, gold_offdiag)`; keep batching, Adam, and the every-10-epoch print.
6. [ ] infrastructure: in `supervised_pytorch_color_mapper.py`, delete `_generate_targets` and remove `projection_targets` from the `train`/`_run_training_loop`/`_train_epoch`/`_train_batch` chain.
7. [ ] infrastructure: implement the class-keyed contrastive/triplet loss in `_compute_combined_loss` (consuming `self._training_labels`) with scale-normalised Lab terms or learnable uncertainty weighting; keep `classification_weight`, clipping, checkpointing, scheduler.
8. [ ] infrastructure: confirm `structured_pytorch_color_mapper.py` is unchanged in this step.
9. [ ] tests: update/replace existing tests that referenced `_generate_targets` (e.g. `test_should_generate_targets` in `tests/colors_of_meaning/infrastructure/ml/test_pytorch_color_mapper.py`) so they assert the new objective.
10. [ ] tests: add the structure-preserving and supervised-objective tests below.
11. [ ] tests: run `tox` for full verification (all 8 gates) and confirm 100% coverage; never rely on `pytest` alone.

## Testing Strategy
Conventions: one logical assertion per test (ML/numerical tests may group related asserts on a single result, e.g. shape plus value range); names `test_should_<behaviour>_when_<condition>`; ML/domain tests use plain `assert`/`pytest.raises` (matching the existing mapper tests, which do not use assertpy); base-entity tests elsewhere use assertpy; mocks/synthetic numpy data only, no network or dataset downloads. Use a fixed seed and `network.eval()` for assertion passes to neutralise dropout until P0-5 lands seeded generators.

New/updated tests (infrastructure ML):
- `test_should_map_near_duplicate_inputs_to_near_identical_lab_when_trained` — train on a batch containing two nearly identical embeddings; assert Delta E between their `embed_to_lab` outputs is below a small threshold (the keystone test that fails under random targets).
- `test_should_lower_structure_loss_when_student_matches_teacher_ordering` — on a synthetic batch with known high/low teacher cosine pairs, assert a teacher-mirroring student similarity yields lower loss than an inverted one.
- `test_should_exclude_self_pairs_when_building_offdiagonal_similarity` — assert the off-diagonal mask drops the diagonal (no self-similarity contribution).
- `test_should_produce_finite_structure_loss_when_batch_is_typical` — assert the loss is finite (no NaN/inf) and non-negative for a representative batch.
- `test_should_remove_generate_targets_when_objective_is_structure_preserving` — assert `_generate_targets` is absent from `PyTorchColorMapper` (e.g. `not hasattr`), replacing the old `test_should_generate_targets`.
- `test_should_train_supervised_with_contrastive_objective_when_labels_set` — set labels, train, assert `embed_to_lab` returns a valid `LabColor` and training completes without the removed `projection_targets` path.
- `test_should_keep_classification_term_influential_when_losses_combined` — assert the normalised/uncertainty-weighted Lab term and the classification term are of comparable magnitude on a synthetic batch (no swamping).
- `test_should_scale_normalise_lab_loss_into_unit_range_when_combining` — assert the normalised Lab loss component lies in the expected bounded range.
- `test_should_remove_generate_targets_from_supervised_mapper_when_refactored` — assert `_generate_targets` is absent from `SupervisedPyTorchColorMapper`.
- Preserve existing valid-range and persistence tests (`embed_to_lab` L/a/b ranges, save/load) — they still hold.

Architectural / boundary tests (pytest-archon):
- `test_should_not_import_torch_in_domain_layer_when_objective_changes` — domain stays pure.
- `test_should_not_import_infrastructure_in_application_layer` — confirm the use case did not gain an infrastructure dependency.

Cross-referenced acceptance (delivered by `005-p0-5-determinism-structure-metric`, asserted there): training reduces the held-out similarity-discrepancy (Spearman) metric versus an untrained projector, and identical seeds reproduce identical Lab outputs. This plan's near-duplicate test is the local proxy; the global metric is P0-5's.

Final verification: `tox` (flake8, black, bandit, semgrep, xenon, radon, mypy, pip-audit) green with 100% coverage. For fast TDD iteration use `tox -- tests/colors_of_meaning/infrastructure/ml/test_pytorch_color_mapper.py`, never bare `pytest`.

## Observability Plan
Keep the existing every-10-epoch loss `print` in both mappers; the reported value now reflects the bounded structure-preserving loss (expect a small magnitude versus the prior O(10^3-10^4)). Do not add comments to explain the new scale — name helpers (`_teacher_similarity`, `_student_similarity`) so intent is self-evident. Structured logging with `correlation-id`, metrics emission, and the structure-preservation correlation report are owned by `005-p0-5-determinism-structure-metric`; this step must remain compatible with them. No secrets logged.

## Risks and Mitigations
- Collapse to a constant color (all-equal Lab can trivially satisfy some similarity targets): mitigate by choosing a similarity that penalises degenerate variance (mean-centred cosine), and by the near-duplicate test plus P0-5's correlation metric catching collapse via low/zero structure preservation.
- Dropout/seed nondeterminism makes the near-duplicate assertion flaky: assert via `embed_to_lab` (already runs `network.eval()`), use a fixed seed, and a tolerance threshold; full determinism arrives with P0-5.
- Degenerate single-row batches yield an empty off-diagonal mask: define and test skip-or-zero behaviour for `batch_size == 1`.
- Supervised multi-task imbalance persists if normalisation is mis-scaled: pin it with the comparable-magnitude and unit-range tests above.
- Complexity gates (`xenon`/`radon`) trip on a larger `train`: extract private helpers to keep methods small.
- `util.cos_sim` import provenance: it is part of the already-declared `sentence-transformers` dependency, so `pip-audit` and `setup.cfg`/`pyproject.toml` need no new entry; if a one-line torch cosine is preferred instead (Open Question), no dependency change either.
- Existing tests referencing `_generate_targets` will break: update them in the same change set (Task 9) so coverage stays meaningful and 100%.
