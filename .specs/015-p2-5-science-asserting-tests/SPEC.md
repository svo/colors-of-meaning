# Feature: Tests That Assert Science, Not Plumbing

## Overview

The suite is large (725 tests) and at 100% coverage, but it almost entirely asserts **plumbing** (shapes, types, value ranges, that a method ran without raising) rather than **science** (that the projector and distance metric preserve and order semantic structure). The most flagrant symptom is a family of vacuous `assert True` statements that pass regardless of behaviour. The ROADMAP cites `tests/colors_of_meaning/infrastructure/ml/test_pytorch_color_mapper.py:68` (inside `test_should_train_mapper`); the same anti-pattern recurs at lines 86, 110, 118 in that file, at lines 59, 77, 103, 111 in `test_structured_pytorch_color_mapper.py`, at lines 152, 263 in `test_supervised_pytorch_color_mapper.py`, and at line 55 in two persistence tests. Under these tests, a projector that emitted uniform noise (which, per ROADMAP §1, is effectively what `_generate_targets` trains toward today) would leave the suite green.

This step makes the test suite a genuine empirical instrument by adding three classes of science-asserting tests and removing the vacuous ones:

1. **Distance ordering** — a `DistanceCalculator` test pinning that moving histogram mass between **perceptually close** codebook colors yields a *smaller* distance than moving it between **perceptually distant** colors. This is the same perceptual-geometry property `001-p0-1-lab-emd-distance` introduces (and that the current 1-D bin-index Wasserstein provably fails); P2-5 makes that property a permanent regression guard rather than a one-off acceptance check.

2. **Structure preservation** — a projector test asserting that embedding-space similarity is reflected in color-space proximity: near-duplicate embeddings land at near-identical Lab colors, and (as a stronger statement) the rank correlation between embedding cosine similarity and color-space distance clears a floor. This directly exercises the metric defined in `005-p0-5-determinism-structure-metric` and the objective in `002-p0-2-structure-preserving-training`; it is the test that would fail if the projector emitted noise.

3. **Accuracy floor** — an end-to-end test on a tiny, seeded, synthetic fixture asserting the color-histogram classifier clears a minimum accuracy meaningfully above chance. This is the unit-test-scale guard for the headline AG News result in `006-p0-6-end-to-end-agnews-table`; it does not reproduce the published numbers (that is P0-6's integration job) but ensures the pipeline is not emitting label-independent garbage.

The step also discharges a NO-COMMENTS debt: `src/colors_of_meaning/infrastructure/evaluation/hnsw_classifier.py` carries explanatory class/method docstrings (lines 13-19, 29-41) and inline comments (lines 63, 72, 75, 84) that violate the house rule; these are removed and, where they carried real intent, that intent is moved into expressive names. The `# noqa: N803` / `# type: ignore` pragmas on lines 24, 43, 52 are tooling directives, not explanatory comments, and are retained.

Per the house "PREFER EDITING over creating" rule and the ROADMAP DoD ("the suite would FAIL if the projector emitted noise"), the existing weak tests are **rewritten in place** to assert observable outcomes; new science tests are added to the same existing test modules rather than introducing parallel files. No production behaviour changes except the comment removal in `hnsw_classifier.py`.

This is a P2 (engineering & honesty) step. It is independent of the P0 science fixes in mechanism, but its assertions are calibrated against them: the structure-preservation and distance-ordering tests express the *properties* P0-1/P0-2/P0-5 deliver, so running P2-5 before those land will (correctly) fail — see Open Questions on sequencing and threshold calibration.

## User Stories

- As a maintainer, I want the test suite to fail when the projector emits semantically structureless colors so that "100% coverage" reflects correctness, not just line execution.
- As a researcher, I want a fast regression test pinning perceptual distance ordering (close colors < distant colors) so that a regression in the `DistanceCalculator` is caught without a full corpus run.
- As a researcher, I want a test asserting that near-duplicate meanings map to near-identical colors so that the blog's central claim is continuously verified, not merely assumed.
- As a researcher, I want a tiny seeded end-to-end accuracy-floor test so that the color pipeline is guarded against label-independent output between full AG News runs.
- As a maintainer, I want every vacuous `assert True` replaced by an assertion on an observable outcome so that no test passes regardless of behaviour.
- As a maintainer, I want `hnsw_classifier.py` free of comments so that it complies with the NO-COMMENTS rule like the rest of the codebase.

## Acceptance Criteria

- [ ] Given two histograms that move equal mass between **perceptually close** codebook colors versus between **perceptually distant** codebook colors, when each distance is computed by the `DistanceCalculator`, then the close-color distance is strictly less than the distant-color distance.
- [ ] Given two identical histograms, when the distance is computed, then the result is approximately zero (regression guard retained alongside the ordering test).
- [ ] Given two embeddings that are near-duplicates (cosine similarity near 1), when both are mapped through a seeded projector, then their resulting Lab colors are within a small Delta E of each other.
- [ ] Given a seeded projector trained (or wired) so that color structure mirrors embedding structure, when the Spearman rank correlation between pairwise embedding cosine similarity and pairwise color-space distance is computed over a synthetic batch, then its magnitude clears a configured floor (a structureless/noise projector would not).
- [ ] Given a tiny, seeded, class-separated synthetic dataset, when the color-histogram classifier is fitted on the train split and predicts the test split, then accuracy clears a minimum floor meaningfully above the chance rate for that fixture.
- [ ] Given the same seed and the same synthetic input, when the projector maps a batch twice, then the two Lab outputs are identical (the determinism precondition that makes the science thresholds stable; coordinates with `005-p0-5-determinism-structure-metric`).
- [ ] Given the current vacuous `assert True` tests, when this step completes, then each is rewritten to assert an observable outcome (e.g. that training reduced loss, that loaded weights reproduce the saved mapper's outputs) and no `assert True` remains in `tests/`.
- [ ] Given `src/colors_of_meaning/infrastructure/evaluation/hnsw_classifier.py`, when this step completes, then it contains no docstrings or explanatory inline comments (only `# noqa` / `# type: ignore` tooling pragmas remain) and its behaviour is unchanged.
- [ ] Given the full quality gate, when `tox` runs, then all 8 gates pass, coverage remains 100%, and every new/rewritten test follows the `test_should_..._when_...` naming pattern with one logical assertion (related numerical asserts on the same result grouped where the house rule permits).

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)
No changes. The science tests consume existing domain types as fixtures and assertion subjects: `domain/model/lab_color.py` (`LabColor`, `to_tuple`, `clamp`), `domain/model/colored_document.py` (`ColoredDocument`, `from_color_sequence`, `histogram`, `num_bins`), `domain/model/color_codebook.py` (`ColorCodebook.colors`, `num_bins`), `domain/model/evaluation_sample.py` (`EvaluationSample`), and the ports `domain/service/distance_calculator.py`, `domain/service/color_mapper.py`, `domain/service/classifier.py`. None are modified.

### Application Layer (`src/colors_of_meaning/application/`)
No changes. The accuracy-floor test exercises `application/use_case/encode_document_use_case.py` (`EncodeDocumentUseCase`) as a collaborator of the color-histogram classifier, via its existing constructor, with no signature change.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- `infrastructure/evaluation/hnsw_classifier.py` — the only production file edited: remove the class docstring (lines 13-19), the `__init__` docstring (lines 29-41), and the inline comments (lines 63, 72, 75, 84). Where a comment named intent (e.g. "Create and initialize HNSW index", "Set query-time search parameters", "Search returns (indices, distances) - note opposite order from FAISS"), fold that intent into expressive local names or small private helpers rather than a comment, keeping behaviour byte-for-byte equivalent. Retain `# noqa: N803` (lines 24, 43) and `# type: ignore` (line 52).
- Test targets under `tests/colors_of_meaning/infrastructure/`: the science assertions and the `assert True` rewrites land in the existing modules `test_wasserstein_distance_calculator.py`, `test_pytorch_color_mapper.py`, `test_structured_pytorch_color_mapper.py`, `test_supervised_pytorch_color_mapper.py`, `test_color_histogram_classifier.py`, and (for the comment-removal regression / behaviour-preservation guard) `test_hnsw_classifier.py`. No new infrastructure production modules.

### Interface Layer (`src/colors_of_meaning/interface/`)
No changes. No CLI or API code is touched by this step.

### Shared Layer
No production changes. The structure-preservation and distance-ordering tests reuse `shared/lab_utils.py:delta_e` (CIE76 Euclidean Lab) as the perceptual yardstick for "close vs distant" and "near-identical colors", giving the test thresholds a single, documented basis. If a shared test-fixture helper for synthetic class-separated embeddings is warranted (see PLAN), it lives under `tests/`, not `shared/`.

## API Contracts
None. No FastAPI endpoints, controllers, or DTOs are added or modified.

## CLI Impact
None. No CLI commands, arguments, or entry points change.

## Dependency Injection
No new wiring. Science tests construct subjects directly with constructor injection in the Arrange step, matching the existing test style: the color-histogram classifier receives a `Mock`/stub embedding adapter, a real `EncodeDocumentUseCase`, and a real `DistanceCalculator` exactly as `test_color_histogram_classifier.py` does today; the projector tests instantiate the mapper with `device="cpu"`. No Lagom container changes; no production DI graph changes. The abstract-to-concrete mappings under test remain `DistanceCalculator -> WassersteinDistanceCalculator`, `ColorMapper -> PyTorchColorMapper` (and structured/supervised variants), `Classifier -> ColorHistogramClassifier`.

## Observability
No production observability changes. The accuracy-floor and structure-preservation tests should, on failure, surface the measured value versus the floor in the assertion message so a regression is diagnosable from CI output without re-running locally; this is a test-readability concern, not a logging/metrics/tracing change. No `correlation-id`, metric, or span is added.

## Open Questions

1. **Numeric thresholds and floors.** What concrete values for: (a) the "near-identical colors" Delta E tolerance for near-duplicate embeddings; (b) the structure-preservation Spearman magnitude floor; (c) the end-to-end accuracy floor above chance on the synthetic fixture? These must be lenient enough to be stable across platforms/seeds yet strict enough that a noise projector fails. Preference: calibrate empirically after `001`/`002`/`005` land, and pin conservative values with the measured margin recorded in the test name/message. (Ties directly to the threshold calibration noted in `001-p0-1-lab-emd-distance` Open Questions and the correlation sign convention in `005-p0-5-determinism-structure-metric`.)
2. **CI vs slow-marked suite for the accuracy-floor test.** Should the end-to-end accuracy-floor test run in the default `tox` gate, or be tagged so it can be excluded from fast runs? `setup.cfg` already defines `markers = integration, benchmark` and the default invocation runs `-m "not benchmark"`. Options: keep it unmarked (tiny synthetic fixture, must stay fast and deterministic — preferred), mark it `integration`, or introduce a new `slow` marker. Whatever is chosen must preserve 100% coverage in the default gate (a marker that excludes a test by default would drop coverage of the lines it uniquely exercises).
3. **Sequencing against P0.** The distance-ordering and structure-preservation assertions encode properties delivered by `001`/`002`/`005`. Run before those land and they fail by design. Should P2-5 be merged after the P0 milestone, or should the not-yet-satisfied assertions be staged (e.g. `xfail` with a strict flag) until P0 lands? Preference: sequence P2-5 after P0-6 so the suite is green, avoiding `xfail` debt.
4. **Synthetic structure-preservation fixture vs trained projector.** Should the structure-preservation test train a small projector on seeded data (slower, exercises `002`'s objective) or assert the property against a deterministically-wired mapper / directly-constructed colors (faster, isolates the metric)? The near-duplicate-colors criterion is cheap and seed-stable; the Spearman-floor criterion may need a short trained projector — decide whether that belongs in the fast gate (Open Question 2).
5. **What observable replaces each `assert True`?** For `test_should_train_mapper` / small-batch / print-loss variants: assert the post-training loss is below the pre-training loss (or below a ceiling) on a seeded batch. For `test_should_load_weights`: assert the reloaded mapper reproduces the saved mapper's `embed_batch_to_lab` output bit-for-bit. Confirm these are the right observables, and that asserting "loss decreased" is not flaky on 1-2 epoch CPU runs (may need a fixed seed and a small but non-trivial epoch count).
6. **Scope of the comment-removal regression guard.** Is a behaviour-preservation assertion in `test_hnsw_classifier.py` (already present and passing) sufficient evidence that the `hnsw_classifier.py` refactor is safe, or should an explicit architectural/no-comment test be added (e.g. asserting the source file contains no `#`-comment lines beyond `noqa`/`type: ignore`)? A repo-wide no-comment lint may already be a `tox` gate concern — reconcile with `016-p2-6-reconcile-docs` rather than duplicating.
