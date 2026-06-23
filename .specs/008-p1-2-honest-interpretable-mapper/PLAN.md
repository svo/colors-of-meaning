# Plan: honest-interpretable-mapper

## Implementation Strategy
Replace the three dishonest target derivations in `infrastructure/ml/structured_pytorch_color_mapper.py` with honest signals, working inside the infrastructure, interface, and shared layers so the `ColorMapper` port, the application use case, and the domain stay untouched. Reuse the exact pattern the supervised mapper already established: optional setters (`set_training_labels`) populate per-sample side information that `train` consumes; this plan adds `set_sentiment_scores` and `set_training_texts` and wires them in `interface/cli/train.py` next to the existing `_configure_supervised_mapper` call.

1. Sentiment-driven lightness: `_derive_lightness_targets` maps an optional per-sample sentiment array monotonically onto `[0, 100]`; with no scores set it returns the documented neutral fallback (constant 50.0, matching the current degenerate branch). IMDB labels (0/1) become two L bands; a continuous polarity maps linearly.
2. Concreteness-driven chroma: `_derive_chroma_targets` looks up each set text's concreteness through a new concreteness-lookup port and maps it monotonically onto `[0, max_chroma]`; with no texts set it returns the neutral fallback (`max_chroma / 2`). Out-of-vocabulary text resolves to a defined neutral concreteness.
3. Semantic hue ordering: `_derive_hue_targets` keeps KMeans on L2-normalised embeddings but replaces `2π·label/K` with angles assigned by a deterministic 1-D ordering of `kmeans.cluster_centers_`, so adjacent ranks (hence adjacent meanings) get adjacent angles.
4. New concreteness lexicon: a `domain/service` port plus an `infrastructure` implementation that loads bundled Brysbaert-style norms via `resources.get_resource_path` and scores a text by token-mean concreteness with a neutral OOV fallback. A bundled data resource lands under `src/colors_of_meaning/resources/`; a tiny fixture lexicon is used in unit tests.
5. Config + wiring: `StructuredMapperConfig` gains the honesty knobs with `from_yaml`/`to_yaml` coverage; `_create_color_mapper` constructs and injects the lexicon, and the structured training path sets sentiment and texts before `train`.

Develop test-first (TDD): for each axis, first write the failing honest-behaviour test against the current proxy implementation (e.g. a more-positive batch must yield higher lightness; a concrete text must yield higher chroma than an abstract one; nearest-centroid clusters must get adjacent hues), confirm it fails, then implement. Cross-references: depends on `001-p0-1-lab-emd-distance`, `002-p0-2-structure-preserving-training`, `003-p0-3-shuffle-stratify-sampling`, `004-p0-4-eval-mapper-coverage-ef-fix`, `005-p0-5-determinism-structure-metric`, `006-p0-6-end-to-end-agnews-table`; enables `010-p1-4-ablations` (interpretability-versus-performance tradeoff).

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
- `domain/service/color_mapper.py`: no change. `ColorMapper` ABC and `QuantizedColorMapper` keep current signatures; honest derivation and the new setters are infrastructure details (precedent: `set_training_labels` is not on the port).
- `domain/service/` (new port, recommended): add a `concreteness_lexicon.py` ABC, e.g. `ConcretenessLexicon` with `score(text: str) -> float`, pure and torch-free. Placement here keeps DI symmetric with `ColorMapper` and lets infrastructure own the file I/O. (Open Question: a `shared/` utility instead.)
- `domain/model/lab_color.py`: no change; honest derivations must keep L in `[0, 100]` and chroma in `[0, max_chroma]` so `__post_init__` never raises and `clamp()` stays a safety net.
- A `pytest-archon` boundary test continues to forbid torch/sklearn/I-O imports in `domain/`, including the new port module.

### Application Layer (`src/colors_of_meaning/application/`)
No changes. `application/use_case/train_color_mapping_use_case.py` already calls `self.color_mapper.train(embeddings=..., epochs=..., learning_rate=...)`; honest targets are derived behind the port and side information is injected by the CLI. A `pytest-archon` test confirms application gains no infrastructure or torch import and does not import the concreteness lexicon implementation.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- `infrastructure/ml/structured_pytorch_color_mapper.py`
  - Constructor: accept an optional `ConcretenessLexicon` (default-constructed to the bundled implementation if `None`) and initialise `self._sentiment_scores: Optional[...] = None` and `self._training_texts: Optional[List[str]] = None`, mirroring the supervised mapper's `self._training_labels`.
  - Add `set_sentiment_scores(scores)` and `set_training_texts(texts)`.
  - Rewrite `_derive_lightness_targets` (lines 221-232): drop `np.mean(embeddings, axis=1)`; if sentiment scores are set, map them monotonically onto `[0, 100]` (binary labels to two bands; continuous to linear); else neutral 50.0. Keep the empty/constant guard.
  - Rewrite `_derive_chroma_targets` (lines 234-245): drop `np.var(embeddings, axis=1)`; if texts are set, compute per-text concreteness via the lexicon and map monotonically onto `[0, max_chroma]`; else neutral `max_chroma / 2`. Keep the constant guard.
  - Rewrite the angle assignment in `_derive_hue_targets` (lines 214-217): compute a deterministic 1-D ordering of `kmeans.cluster_centers_` (extract `_order_clusters_by_centroid`) and map each label to an angle by its rank; keep KMeans `random_state=42`, normalisation, and the `num_clusters = min(self.num_clusters, len(embeddings))` clamp.
  - Extract small private helpers (`_sentiment_to_lightness`, `_concreteness_to_chroma`, `_order_clusters_by_centroid`) to keep cyclomatic complexity within `xenon`/`radon` limits and avoid comments.
- `infrastructure/ml/brysbaert_concreteness_lexicon.py` (new): implement `ConcretenessLexicon`; load bundled norms once via `resources.get_resource_path(...)`, tokenise a text, average in-lexicon token concreteness, return a defined neutral value for fully-OOV text. No network call. (Placement under `infrastructure/ml/` proposed; `infrastructure/dataset/` is an alternative — Open Question.)
- `src/colors_of_meaning/resources/` (new data file): bundle the concreteness norms (full, trimmed subset, or placeholder pending licensing — Open Question). Shipped automatically via `include_package_data = True` and the existing `resources` package mapping in `setup.cfg`.

### Interface Layer (`src/colors_of_meaning/interface/`)
- `interface/cli/train.py`
  - `_create_color_mapper` (lines 57-95): in the `structured` branch construct the concreteness lexicon and pass it into `StructuredPyTorchColorMapper`; thread the new `StructuredMapperConfig` fields.
  - `main`/`_execute_training` (lines 122-165): on the structured path, set sentiment scores (from IMDB labels when the dataset provides them via `_load_supervised_data`, else per the configured fallback/source) and set training texts before training — mirroring `_configure_supervised_mapper` (lines 114-119). Optionally add a `--sentiment-source` field on `TrainArgs` mapped to `StructuredMapperConfig.sentiment_source` (Open Question).
  - No new top-level command; no API change.

### Shared Layer (`src/colors_of_meaning/shared/`)
- `shared/synesthetic_config.py`: extend `StructuredMapperConfig` (lines 46-51) with `sentiment_source`, `concreteness_resource`, and any band/scale parameters; round-trip them in `from_yaml` (lines 76-92) and `to_yaml` (lines 94-107). Defaults preserve current neutral-fallback behaviour. `shared` stays torch-free; the resource path is resolved via the existing `resources` helper, not added here. `shared/lab_utils.py` unchanged.

## Dependency Injection
No Lagom container change (training is CLI-driven). The concreteness lexicon is a `domain/service` port; its concrete implementation is constructed in `_create_color_mapper` and constructor-injected into `StructuredPyTorchColorMapper` (default-constructed when omitted, so existing direct test instantiations still work). Sentiment scores and texts are injected via setters at wiring time, exactly as `set_training_labels` is today. If a future API path needs the lexicon, the existing `Container()` in `interface/api/main.py` can bind the port to its implementation without touching consumers.

## Task List
1. [ ] domain: confirm `domain/service/color_mapper.py` needs no signature change; record the no-change decision (port stays `train(embeddings, epochs, learning_rate)`).
2. [ ] domain: add the `ConcretenessLexicon` ABC in `domain/service/concreteness_lexicon.py` (pure, `score(text) -> float`), with `__init__.py` present.
3. [ ] application: confirm `application/use_case/train_color_mapping_use_case.py` needs no change (behaviour and side-info injection sit behind the port / in the CLI).
4. [ ] infrastructure: add the bundled concreteness norms data file under `src/colors_of_meaning/resources/` (placeholder pending licensing) and a tiny test fixture lexicon.
5. [ ] infrastructure: implement `BrysbaertConcretenessLexicon` in `infrastructure/ml/brysbaert_concreteness_lexicon.py` loading via `resources.get_resource_path`, with token-mean scoring and a neutral OOV fallback; no network call.
6. [ ] infrastructure: add `set_sentiment_scores`, `set_training_texts`, optional `ConcretenessLexicon` constructor param, and the `_sentiment_scores`/`_training_texts` fields to `StructuredPyTorchColorMapper`.
7. [ ] infrastructure: rewrite `_derive_lightness_targets` to map sentiment monotonically onto `[0, 100]` with neutral fallback; extract `_sentiment_to_lightness`.
8. [ ] infrastructure: rewrite `_derive_chroma_targets` to map per-text concreteness monotonically onto `[0, max_chroma]` with neutral fallback; extract `_concreteness_to_chroma`.
9. [ ] infrastructure: replace `2π·label/K` in `_derive_hue_targets` with rank angles from `_order_clusters_by_centroid(kmeans.cluster_centers_)`; keep `random_state=42` and the cluster clamp.
10. [ ] shared: extend `StructuredMapperConfig` with the honesty fields and round-trip them in `from_yaml`/`to_yaml`.
11. [ ] interface: in `interface/cli/train.py`, construct and inject the lexicon in `_create_color_mapper` and set sentiment scores + texts on the structured path before training.
12. [ ] interface: optionally add `--sentiment-source` on `TrainArgs` mapped to config (gated by the Open Question).
13. [ ] tests: rewrite `TestStructuredTargetDerivation` (lightness==mean, chroma==variance, hue `2π·label/K`) to assert the honest behaviour, keeping coverage at 100%.
14. [ ] tests: add the infrastructure, lexicon, config, and architectural tests below.
15. [ ] tests: run `tox` for full verification (all 8 gates) and confirm 100% coverage; never rely on `pytest` alone.

## Testing Strategy
Conventions: one logical assertion per test (ML/numerical tests may group related asserts on a single result, e.g. shape plus range); names `test_should_<behaviour>_when_<condition>`; ML/domain tests use plain `assert`/`pytest.raises` (matching the existing mapper tests, which do not use assertpy); base-entity/config tests may use assertpy. Mocks and synthetic numpy/text data only — no network, no dataset download, no full-norms parsing in the fast suite (use the tiny fixture lexicon). Use a fixed seed and `network.eval()` (via `embed_to_lab`) where dropout could perturb assertions, until P0-5's seeded generators land.

Lightness (sentiment):
- `test_should_increase_lightness_when_sentiment_is_more_positive` — two batches differing only in (higher) sentiment yield higher mean lightness.
- `test_should_map_negative_and_positive_labels_to_distinct_bands_when_sentiment_is_binary` — IMDB-style 0/1 labels produce a low-L band and a high-L band, all in `[0, 100]`.
- `test_should_return_neutral_lightness_when_no_sentiment_is_set` — with no scores set, lightness is the documented neutral fallback.
- `test_should_keep_lightness_in_valid_range_when_sentiment_is_extreme` — extreme sentiment inputs stay within `[0, 100]`.
- `test_should_not_use_embedding_mean_for_lightness_when_sentiment_is_set` — sentiment-driven lightness differs from the old mean-based value on a batch where they would diverge.

Chroma (concreteness):
- `test_should_give_higher_chroma_to_concrete_text_than_abstract_text_when_scored` — concrete versus abstract text against the fixture lexicon.
- `test_should_return_neutral_chroma_when_text_is_out_of_vocabulary` — fully-OOV text yields the midpoint chroma, no NaN/error.
- `test_should_return_neutral_chroma_when_no_text_is_set` — with no texts set, chroma is `max_chroma / 2`.
- `test_should_keep_chroma_in_valid_range_when_concreteness_is_extreme` — values stay within `[0, max_chroma]`.

Hue (semantic ordering):
- `test_should_assign_adjacent_hues_to_nearest_centroid_clusters_when_ordering_by_centroid` — nearest-centroid clusters get closer hue angles than a far-apart pair (fails under `2π·label/K`).
- `test_should_assign_hue_angles_within_circle_when_ordering_clusters` — all angles in `[0, 2π)`, shape `(n, 1)`.
- `test_should_order_clusters_deterministically_when_seed_is_fixed` — repeated derivation yields identical ordering under fixed seed.

Concreteness lexicon (infrastructure):
- `test_should_load_norms_from_bundled_resource_when_constructed` — constructs via `resources.get_resource_path` with the fixture, no network.
- `test_should_average_in_lexicon_token_scores_when_scoring_text` — token-mean over known words.
- `test_should_return_neutral_score_when_all_tokens_are_unknown` — defined OOV fallback.

Config (shared):
- `test_should_round_trip_sentiment_source_when_serialising_structured_config` — `to_yaml`/`from_yaml` preserves `sentiment_source` (assertpy acceptable).
- `test_should_round_trip_concreteness_resource_when_serialising_structured_config` — preserves `concreteness_resource`.
- `test_should_default_structured_config_to_neutral_when_unspecified` — defaults preserve current behaviour.

Interface (CLI wiring):
- `test_should_inject_concreteness_lexicon_into_structured_mapper_when_creating_mapper` — `_create_color_mapper` returns a structured mapper holding a lexicon.
- `test_should_set_sentiment_scores_on_structured_mapper_when_dataset_has_labels` — structured path sets sentiment from labels (mock embedding/dataset; no network).

Architectural / boundary (pytest-archon):
- `test_should_not_import_torch_in_domain_layer_when_lexicon_port_added` — domain (incl. new port) stays pure.
- `test_should_not_import_infrastructure_in_application_layer` — application gains no infrastructure/lexicon import.
- `test_should_define_concreteness_lexicon_interface_in_domain` — the port is an ABC in `domain/service`.

Backward compatibility / regression:
- `test_should_train_structured_mapper_when_no_sentiment_or_text_is_set` — existing embeddings-only call still trains (neutral fallbacks).
- Preserve existing valid-range, save/load, small-batch, and print-cadence tests — they still hold.

Final verification: `tox` (flake8, black, bandit, semgrep, xenon, radon, mypy, pip-audit) green with 100% coverage. For fast TDD iteration use `tox -- tests/colors_of_meaning/infrastructure/ml/test_structured_pytorch_color_mapper.py`, never bare `pytest`.

## Observability Plan
Keep the every-10-epoch loss `print` (lines 111-112). At structured-mapper setup, log a single line naming the active sentiment source and concreteness resource (no secrets, no raw text). Use self-documenting names (`set_sentiment_scores`, `_concreteness_to_chroma`, `_order_clusters_by_centroid`) instead of comments. Structured logging with `correlation-id`, metrics, and the interpretability/structure metrics are owned by `005-p0-5-determinism-structure-metric` and consumed by `010-p1-4-ablations`; remain compatible with them. No secrets logged.

## Risks and Mitigations
- Honesty costs accuracy: constraining axes to sentiment/concreteness/semantic-order may lower the structured mapper's AG News number versus unconstrained. This is the expected, measurable tradeoff; do not hide it — `010-p1-4-ablations` quantifies it. Mitigate surprise by documenting the expectation in the run output and keeping the unconstrained baseline intact.
- Concreteness norms licensing/size: bundling the full Brysbaert list may exceed redistribution terms or bloat the package. Mitigate by shipping a placeholder/trimmed lexicon plus a tiny test fixture now, and confirming the licence before shipping the full file (Open Question); keep tests offline regardless.
- Sentiment source coverage gap: only labelled sentiment datasets (IMDB) have gold sentiment; AG News / free text fall back to neutral or a lightweight offline polarity. Mitigate by making the source configurable and defaulting to labels-when-present; avoid a heavy sentiment model that would download weights and break offline tests.
- IMDB single-class splits from the `max_samples` head-slice (`imdb_dataset_adapter.py` lines 17-19) would make sentiment-driven lightness uniform and the band test meaningless. Mitigate by depending on `003-p0-3-shuffle-stratify-sampling` for stratified sampling; unit tests use synthetic balanced sentiment so they do not depend on the dataset fix.
- New dependency `pip-audit` surface: prefer a pure-data resource over a new sentiment package; if a package is unavoidable, declare it in `setup.cfg` `install_requires` (the runtime-dep source of truth; `pyproject.toml` declares no runtime deps) and verify `pip-audit` stays green.
- Determinism of hue ordering and KMeans: ordering must be reproducible. Mitigate with `random_state=42`, a deterministic 1-D ordering (PCA projection of centroids), and a fixed-seed ordering test; full determinism reinforced by `005-p0-5-determinism-structure-metric`.
- Complexity gates (`xenon`/`radon`) tripping on enlarged derivations: extract `_sentiment_to_lightness`, `_concreteness_to_chroma`, `_order_clusters_by_centroid` to keep methods small.
- Existing `TestStructuredTargetDerivation` tests assert the old proxy shapes and will break: rewrite them in the same change set (Task 13) so coverage stays meaningful and 100%.
- Layer-boundary slip: the concreteness file I/O must live in infrastructure, not the domain port or `shared`; enforce via the architectural tests above.
