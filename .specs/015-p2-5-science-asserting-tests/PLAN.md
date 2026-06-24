# Plan: tests-that-assert-science-not-plumbing

## Implementation Strategy

This is a test-centric step. The production footprint is a single file (`src/colors_of_meaning/infrastructure/evaluation/hnsw_classifier.py`, comment removal only); everything else lands under `tests/`. Work proceeds in four moves, each verified with `tox` (never `pytest` alone):

1. **Discharge the comment debt** in `hnsw_classifier.py` first (smallest, lowest-risk, unblocks a clean NO-COMMENTS suite), relying on the existing `test_hnsw_classifier.py` mocks to prove behaviour is unchanged.
2. **Replace the vacuous `assert True`** tests in place with assertions on observable outcomes (loss decreased after training; reloaded weights reproduce outputs; persistence round-trips a real value). PREFER EDITING: rewrite, do not add a parallel test.
3. **Add the three science assertions** — distance ordering, structure preservation, accuracy floor — into the existing test modules, each as a `test_should_..._when_...` with one logical assertion (grouped numerical asserts on the same result where the house rule allows).
4. **Calibrate thresholds** conservatively and confirm a noise/structureless projector would fail them (the ROADMAP DoD), then run the full gate.

Because the science assertions encode properties that `001`/`002`/`005` deliver, this step is sequenced **after the P0 milestone** (see Risks); the thresholds are pinned against the values those steps produce.

All randomness in tests is seeded (`torch.manual_seed`, `np.random.seed`, and a seeded `torch.Generator` where the mapper accepts one per `005-p0-5-determinism-structure-metric`) so thresholds are stable across runs. Fixtures are tiny synthetic arrays — no network, no dataset download, no real sentence-transformers model — keeping the suite fast and deterministic per the house mocking rules.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
No changes. Domain types are consumed as fixtures/subjects only (`LabColor`, `ColoredDocument`, `ColorCodebook`, `EvaluationSample`, and the `DistanceCalculator` / `ColorMapper` / `Classifier` ports).

### Application Layer (`src/colors_of_meaning/application/`)
No changes. `EncodeDocumentUseCase` is used unmodified as a collaborator in the accuracy-floor test.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
One production edit: `src/colors_of_meaning/infrastructure/evaluation/hnsw_classifier.py` — remove the class docstring (lines 13-19), the `__init__` docstring (lines 29-41), and the inline comments (lines 63, 72, 75, 84); fold any real intent into expressive names (e.g. extract index construction / query-parameter setting into clearly-named private helpers if that reads better than a bare comment) without altering behaviour. Keep `# noqa: N803` (24, 43) and `# type: ignore` (52). All other changes are under `tests/colors_of_meaning/infrastructure/ml/` and `tests/colors_of_meaning/infrastructure/evaluation/`.

### Interface Layer (`src/colors_of_meaning/interface/`)
No changes.

### Shared Layer (`src/colors_of_meaning/shared/`)
No production changes. Tests reuse `shared/lab_utils.py:delta_e` as the perceptual yardstick. Any shared synthetic-embedding fixture helper lives under `tests/`, not `shared/`.

## Dependency Injection

No production DI changes. Test subjects are built with constructor injection in Arrange, mirroring existing tests: the color-histogram classifier takes a stubbed embedding adapter, a real `EncodeDocumentUseCase`, and a real `DistanceCalculator`; mappers are constructed with `device="cpu"`. No new packages, so `pip-audit` is unaffected.

## Task List

1. [ ] tests: characterize current `hnsw_classifier.py` behaviour via the existing `test_hnsw_classifier.py` (fit/predict/majority-vote/encode) and confirm it is green as a safety net before refactoring.
2. [ ] tests: remove docstrings and inline comments from `src/colors_of_meaning/infrastructure/evaluation/hnsw_classifier.py`, folding intent into expressive names; keep `noqa`/`type: ignore`; verify `test_hnsw_classifier.py` still passes (behaviour unchanged) and `flake8`/`black`/`mypy` gates are green.
3. [ ] tests: rewrite `test_pytorch_color_mapper.py::test_should_train_mapper` (line 68) to assert the seeded post-training loss is below the pre-training loss (or a ceiling) instead of `assert True`.
4. [ ] tests: rewrite `test_pytorch_color_mapper.py::test_should_load_weights` (line 86) to assert the reloaded mapper reproduces the saved mapper's `embed_batch_to_lab` output for a fixed input.
5. [ ] tests: rewrite the remaining `assert True` cases in `test_pytorch_color_mapper.py` (lines 110, 118 — small-batch and print-loss-every-10-epochs) to assert an observable training effect rather than mere completion.
6. [ ] tests: rewrite the `assert True` cases in `test_structured_pytorch_color_mapper.py` (lines 59, 77, 103, 111) — training effect and weight-reload-reproduces-output.
7. [ ] tests: rewrite the `assert True` cases in `test_supervised_pytorch_color_mapper.py` (lines 152, 263) — training effect and projector-weight-reload-reproduces-output.
8. [ ] tests: rewrite the persistence `assert True` cases (`test_file_color_codebook_repository.py:55`, `test_in_memory_color_codebook_repository.py:55`) to assert the round-tripped codebook equals the saved one.
9. [ ] tests: add distance-ordering test to `test_wasserstein_distance_calculator.py` — build a small codebook with two perceptually-close colors and one perceptually-distant color (Delta E ranked via `delta_e`), and assert close-color mass-move distance < distant-color mass-move distance. (Cross-ref `001-p0-1-lab-emd-distance` AC: close < distant.)
10. [ ] tests: keep/strengthen the identical-histograms approximately-zero assertion as the ordering test's companion regression guard.
11. [ ] tests: add structure-preservation "near-duplicate -> near-identical color" test in `test_pytorch_color_mapper.py` — map two near-duplicate seeded embeddings and assert their Lab colors are within a small Delta E. (Cross-ref `002-p0-2-structure-preserving-training`.)
12. [ ] tests: add structure-preservation Spearman-floor test — over a seeded synthetic batch, assert the magnitude of the rank correlation between embedding cosine similarity and color-space distance clears the floor (would fail for noise). (Cross-ref `005-p0-5-determinism-structure-metric`.)
13. [ ] tests: add determinism test — same seed + same input -> identical Lab outputs across two mapper runs (precondition for stable thresholds; cross-ref `005`).
14. [ ] tests: add end-to-end accuracy-floor test in `test_color_histogram_classifier.py` — fit on a tiny seeded class-separated synthetic train split, predict the test split, assert accuracy >= floor above chance. (Cross-ref `006-p0-6-end-to-end-agnews-table`.)
15. [ ] tests: calibrate the Delta E tolerance, Spearman floor, and accuracy floor to conservative, platform-stable values; encode the measured margin in each assertion message; confirm a deliberately-structureless projector/distance fails each new test (manual spike, not committed).
16. [ ] tests: confirm naming (`test_should_..._when_...`), one-logical-assertion compliance, and no remaining `assert True` anywhere via `grep -rn "assert True" tests/`.
17. [ ] tests: run `tox` (all 8 gates) and confirm green with 100% coverage; run `tox -e format` if needed.

## Testing Strategy

This step *is* the testing strategy; the following are the science assertions, their fast/deterministic construction, and where they live.

**A. Distance ordering (`test_wasserstein_distance_calculator.py`; ties to `001-p0-1-lab-emd-distance`).**
Construct a small `ColorCodebook` of a handful of `LabColor`s where two are perceptually close and one is far, with the close/far relationship established via `shared/lab_utils.py:delta_e` so the fixture's premise is explicit and checkable. Build two `ColoredDocument` pairs (via `ColoredDocument.from_color_sequence` or direct normalized histograms): one that moves unit mass between the close colors, one between the distant colors. Assert `distance_close < distance_far`. This is a property the legacy 1-D bin-index Wasserstein fails and the POT Lab-EMD of `001` satisfies; keeping it here turns `001`'s acceptance check into a standing regression guard. The companion identical-histogram-approximately-zero test stays. Fast: pure-numpy, no model.

**B. Structure preservation (`test_pytorch_color_mapper.py`; ties to `002` and `005`).**
Two assertions, escalating in strength:
- *Near-duplicate -> near-identical color*: take a seeded base embedding and a tiny-perturbation copy (cosine similarity near 1), map both with a seeded `device="cpu"` mapper, and assert `delta_e(color_a, color_b)` is below a small tolerance. Cheap and seed-stable; it is the minimal expression of "similar meanings map to similar colors".
- *Spearman floor*: over a seeded synthetic batch whose pairwise embedding cosine similarities span a range, compute color-space distances between the mapped Lab colors and assert the magnitude of the rank correlation (embedding-similarity vs color-distance) clears a floor. A noise projector yields ~0 correlation and fails. This reuses the metric idea from `005`; whether it needs a short trained projector or can assert against a deterministically-structured mapper is Open Question 4 in SPEC — default to the cheapest variant that still distinguishes signal from noise.

**C. Accuracy floor (`test_color_histogram_classifier.py`; ties to `006-p0-6-end-to-end-agnews-table`).**
Build a tiny seeded synthetic dataset with clearly separable per-class structure (e.g. class-conditional embedding clusters returned by a stub `encode_document_sentences`), fit `ColorHistogramClassifier` (real `EncodeDocumentUseCase`, real `DistanceCalculator`, stubbed adapter, `hnswlib` index over the small set), predict the held-out split, and assert accuracy >= a floor comfortably above the fixture's chance rate. This is the unit-scale guard for P0-6's headline number: it does not reproduce AG News (that is P0-6's reproducible-command integration job), only ensures the pipeline produces label-correlated output. Determinism via `random_seed=100` (already set in the index) plus seeded synthetic data; speed via a handful of samples.

**D. Replacing `assert True` (multiple `ml/` and `persistence/` modules).**
Each becomes an outcome assertion: training tests assert a seeded loss decrease (or below-ceiling) by capturing loss before/after; weight-load tests assert the reloaded mapper reproduces `embed_batch_to_lab` output for a fixed input (proving the state actually transferred); persistence tests assert the round-tripped codebook equals the saved one. These are the "would the suite notice if it broke?" guards the ROADMAP demands.

**Property-based angle.** Distance ordering and "near-duplicate -> near color" are natural properties; if the `property-based-testing` posture is adopted, a small Hypothesis strategy over Lab triples could assert *monotonicity* (larger embedding distance -> not-smaller color distance, within tolerance) and metric axioms (symmetry, identity-of-indiscernibles) for the `DistanceCalculator`. Kept optional and bounded (fixed `max_examples`, seeded) so the gate stays fast; example-based tests remain the primary guard.

**Keeping it fast and deterministic.** No network, no real embedding model, no dataset adapter I/O — synthetic numpy arrays and stubbed adapters only (house rule: no network calls in unit tests). Every stochastic subject is seeded. Epoch counts are the minimum that still produces a measurable training effect. Thresholds are pinned conservatively (Open Question 1) with the noise-projector falsification checked manually during calibration (task 15).

**Coverage.** Rewriting `assert True` tests preserves the lines they already covered; the new science tests add assertions, not new production lines (except the `hnsw_classifier.py` comment removal, which only deletes lines). 100% coverage is therefore maintained in the default `tox` gate; any marker decision (Open Question 2) must not exclude a test that uniquely covers a line by default.

## Observability Plan

No production observability changes. Test-side only: each science assertion's failure message includes the measured value and the floor/tolerance so a CI failure is diagnosable without a local re-run. No logging, metrics, tracing, or `correlation-id` work in this step.

## Risks and Mitigations

- **Risk: assertions fail because P0 science is not yet merged.** The distance-ordering, structure-preservation, and accuracy-floor tests encode properties from `001`/`002`/`005`/`006`. *Mitigation:* sequence P2-5 after the P0 milestone (SPEC Open Question 3); do not introduce `xfail` debt.
- **Risk: flaky thresholds across platforms/seeds.** Delta E tolerance, Spearman floor, and accuracy floor could pass on one machine and fail on another. *Mitigation:* seed everything; pin conservative floors with a recorded margin; calibrate empirically after P0 lands (task 15) and falsify against a noise projector to confirm the gap.
- **Risk: "loss decreased" is noisy on 1-2 epoch CPU runs.** *Mitigation:* fix the seed and use the smallest epoch count that yields a stable decrease, or assert below a ceiling rather than strict monotonic decrease; choose per Open Question 5.
- **Risk: refactoring `hnsw_classifier.py` silently changes behaviour.** *Mitigation:* the existing `test_hnsw_classifier.py` (fit/predict/majority-vote/encode, with mocked `hnswlib.Index`) is the safety net; run it before and after, change only comments and names, keep the call sequence to `init_index`/`add_items`/`set_ef`/`knn_query` identical.
- **Risk: a new marker drops default-gate coverage below 100%.** *Mitigation:* prefer keeping the accuracy-floor test unmarked in the fast gate (tiny synthetic fixture); only mark if it can be excluded without losing unique line coverage (Open Question 2).
- **Risk: scope creep into duplicating a repo-wide no-comment lint.** *Mitigation:* rely on existing gates plus the behaviour-preservation test for `hnsw_classifier.py`; defer any global no-comment enforcement to `016-p2-6-reconcile-docs` (SPEC Open Question 6).
