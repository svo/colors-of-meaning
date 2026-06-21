# Feature: Structure-Preserving Training Objective

## Overview
The two learnable projectors (`PyTorchColorMapper` and `SupervisedPyTorchColorMapper`) are currently trained to regress onto **uniform-random** Lab targets produced by `_generate_targets`, so the headline thesis — *similar meanings map to similar colors* — is never optimised for and the resulting color method is decorative. This feature deletes the random-target objective and replaces it with a structure-preserving loss: per batch we build a frozen-teacher similarity target from the 384-dim embeddings (`util.cos_sim(emb, emb).detach()`), build a student similarity from the 3-D Lab outputs, and minimise the MSE between their off-diagonal entries — the structure-preserving analogue of sentence-transformers `CosineSimilarityLoss`. The supervised variant replaces its dominant random-target MSE with a class-keyed contrastive objective and a normalised (or learnable-uncertainty-weighted) multi-task sum so the cross-entropy term is no longer swamped (today Lab MSE is O(10^3-10^4) versus 0.1*CE at O(1)). Success is measured by P0-5's held-out similarity-discrepancy metric and a near-duplicate co-location test that would fail under random targets.

## User Stories
- As a researcher validating the Colors of Meaning thesis, I want the projector trained so that semantically similar texts receive perceptually similar Lab colors, so that the AG News color-method number reflects a real mechanism rather than noise.
- As a researcher, I want two near-duplicate inputs to land at near-identical Lab colors after training, so that I have direct evidence the projector preserves semantic structure.
- As a maintainer of the supervised mapper, I want the classification signal to meaningfully shape the color space instead of being drowned by an unrelated MSE term, so that class-keyed structure is actually learned.
- As a pipeline owner, I want the structure-preserving loss to decrease a held-out structure-discrepancy metric during training, so that P0-6's end-to-end run is trustworthy.

## Acceptance Criteria
- [ ] Given the unconstrained `PyTorchColorMapper`, when training runs, then `_generate_targets` is no longer invoked and the optimiser minimises an MSE between student off-diagonal Lab similarities and the detached teacher off-diagonal embedding cosine similarities.
- [ ] Given two near-duplicate embeddings, when the trained `PyTorchColorMapper` maps them to Lab, then the Delta E between the two colors is below a small threshold (a test that would fail under random targets).
- [ ] Given two embeddings with high teacher cosine similarity and two with low teacher cosine similarity, when the structure-preserving loss is computed on a synthetic batch, then a student that mirrors the teacher ordering yields a lower loss than one that inverts it.
- [ ] Given the supervised `SupervisedPyTorchColorMapper`, when training runs, then the random-target MSE term is removed and replaced by a class-label-keyed contrastive (or triplet) objective on the Lab outputs.
- [ ] Given the supervised multi-task loss, when the Lab term and classification term are combined, then the Lab term is scale-normalised (L into [0,1], a/b into [-1,1]) or weighted by learnable uncertainty so the classification term is not dominated.
- [ ] Given a configured `seed` and identical inputs, when the structure-preserving loss is evaluated, then it is deterministic and bounded (no NaN/inf), with the off-diagonal mask excluding self-similarities.
- [ ] Given a held-out evaluation set, when training completes, then P0-5's similarity-discrepancy metric is reduced relative to an untrained projector (cross-referenced acceptance; the metric itself is delivered by `005-p0-5-determinism-structure-metric`).
- [ ] Given the full suite, when `tox` runs, then all gates pass, coverage stays 100%, every new test has one logical assertion and is named `test_should_..._when_...`, and no comments are introduced.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)
No changes. The `ColorMapper` port in `domain/service/color_mapper.py` keeps its existing signatures (`embed_to_lab`, `embed_batch_to_lab`, `train(embeddings, epochs, learning_rate)`, `save_weights`, `load_weights`); the structure-preserving objective is an internal implementation detail of the infrastructure adapters and requires no port change. `QuantizedColorMapper` is unaffected. Domain stays pure (no torch). The existing `delta_e` (CIE76) helper lives in `shared/lab_utils.py`, not the domain, and remains the reference definition of perceptual distance.

### Application Layer (`src/colors_of_meaning/application/`)
No changes. `TrainColorMappingUseCase.execute(...)` already delegates to `color_mapper.train(embeddings=..., epochs=..., learning_rate=...)`; because the objective changes behind the port, no orchestration change is required. Determinism seeding and structure-metric checkpointing are introduced by `005-p0-5-determinism-structure-metric`, which owns any application/use-case bootstrap edits.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
Primary scope.
- `infrastructure/ml/pytorch_color_mapper.py`: delete `_generate_targets`; replace the random-target MSE in `train` with a structure-preserving objective. Per batch: `gold = util.cos_sim(batch_embeddings, batch_embeddings).detach()` (frozen teacher); build a student Lab similarity from the network's 3-D outputs (e.g. `1 - normalized Delta E`, or cosine of mean-centred Lab); `loss = MSE(student_offdiag, gold_offdiag)` over the strict upper/lower triangle (self-pairs excluded). `LabProjectorNetwork` is unchanged (its sigmoid/tanh Lab head already bounds outputs to valid ranges).
- `infrastructure/ml/supervised_pytorch_color_mapper.py`: delete `_generate_targets`; in `_compute_combined_loss` replace `projection_loss = mse_loss(lab_output, projection_targets)` with a class-keyed `ContrastiveLoss`/`TripletMarginLoss` on `lab_output` using `self._training_labels`; either scale-normalise the Lab term (L/100 into [0,1], a/b/127.5 into [-1,1]) or introduce learnable log-variance uncertainty weights so `classification_weight * cross_entropy` is not swamped. Update `_run_training_loop`/`_train_epoch`/`_train_batch`/`_compute_combined_loss` signatures to stop threading `projection_targets`. Checkpoint-on-improvement, gradient clipping, AdamW, and cosine LR schedule are retained.
- `infrastructure/ml/structured_pytorch_color_mapper.py`: no functional change in this step (its loss is already structure-shaped via hue/lightness/chroma targets); honesty of those axes is `008-p1-2-honest-interpretable-mapper`.
- Uses `sentence_transformers.util.cos_sim` (a torch utility) and torch only — both already declared in `setup.cfg`/`pyproject.toml` (`sentence-transformers`, `torch`). No new dependency is expected.

### Interface Layer (`src/colors_of_meaning/interface/`)
No changes. `interface/cli/train.py` already constructs each mapper via `_create_color_mapper(args, config)` and injects it into `TrainColorMappingUseCase`; the supervised path already calls `set_training_labels(labels)` before training. The new contrastive objective consumes those same already-set labels, so no CLI flag or wiring change is required.

### Shared Layer
No changes required by this step. `shared/lab_utils.delta_e` (numpy-only CIE76) is available as the reference perceptual distance; if the student similarity reuses a Delta E formulation it will be reimplemented in torch inside the infrastructure adapter (the shared helper operates on `LabColor` scalars, not batched tensors, and importing torch into `shared` is avoided to keep it dependency-light). `TrainingConfig.seed` (42) and `SupervisedMapperConfig.classification_weight`/`num_classes` are consumed unchanged; any new structure-loss hyperparameter (e.g. contrastive margin) is an Open Question for config placement.

## API Contracts
No changes. This step introduces no new endpoints and modifies no DTOs. The training objective is reached only via the `train` CLI command, not the FastAPI surface.

## CLI Impact
No new flags or commands. `train.py` continues to support `--mapper-type {unconstrained,structured,supervised}`. Observable behaviour change: the printed per-epoch loss now reflects a structure-preserving objective (a small bounded similarity-MSE) rather than the previous O(10^3-10^4) random-target MSE; documentation/log expectations should account for the new scale. The `--mapper-type` reach fix for `eval` is out of scope here (owned by `004-p0-4-eval-mapper-coverage-ef-fix`).

## Dependency Injection
Unchanged and consistent with project reality. CLI wires mappers through explicit factory functions (`_create_color_mapper` in `interface/cli/train.py`) and constructor-injects them into `TrainColorMappingUseCase`; the API runtime builds a Lagom `Container()` in `interface/api/main.py` but does not touch training. No new ports or container registrations are needed because the change is internal to existing `ColorMapper` implementations resolved behind the existing port.

## Observability
Preserve the existing per-epoch `print` cadence (every 10 epochs) in both mappers; the logged value now represents the structure-preserving loss. Structured logging with `correlation-id`, metrics, and the structure-preservation correlation report are introduced by `005-p0-5-determinism-structure-metric` (which adds the held-out metric used for checkpoint selection); this step must not regress those once they land. No secrets are logged.

## Open Questions
- Student similarity definition: `1 - normalized Delta E` versus cosine of mean-centred Lab — which yields the most stable gradients and the best correlation under P0-5's metric? (ROADMAP lists both as acceptable; default proposed: cosine of mean-centred Lab for scale-invariance, with Delta E variant tested.)
- Supervised contrastive choice: `ContrastiveLoss` versus `TripletMarginLoss`, and fixed-margin versus learnable-uncertainty multi-task weighting — which keeps the classification term influential without destabilising training?
- Hyperparameter placement: should the contrastive margin / uncertainty-weighting toggle / student-similarity mode live in `TrainingConfig` or `SupervisedMapperConfig` in `shared/synesthetic_config.py`, or default-only in the adapter constructor? (Adding config fields requires `from_yaml`/`to_yaml` round-trip coverage.)
- Minimum batch size for a meaningful teacher similarity matrix: with `batch_size = min(32, len(embeddings))`, single-row batches make the off-diagonal mask empty — define the degenerate-batch behaviour (skip versus zero-loss) and its test.
- Determinism boundary: this step assumes P0-5 supplies seeded `torch.Generator`/`torch.manual_seed`. Until P0-5 lands, should the near-duplicate co-location test tolerate dropout/seed noise via a threshold, or set `network.eval()` for the assertion pass? (Proposed: assert on `embed_to_lab`, which already calls `eval()`.)
- Whether reusing `sentence_transformers.util.cos_sim` (versus a one-line torch cosine) is preferred for provenance/readability given it pulls the symbol from an already-installed dependency.
