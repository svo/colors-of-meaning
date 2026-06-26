# Plan: Authored document corpus source from the filesystem

## Implementation Strategy

Make a directory of named documents a first-class text source by implementing the
**existing** `DatasetRepository` port with a new
`DocumentCorpusDatasetAdapter`, so every consumer (`EvaluateUseCase`,
`RateDistortionSweepUseCase`, `AblationSweepUseCase`, `EvaluationSuiteUseCase`,
`visualize_corpus`) works unchanged with the author as the class label. The work is
a directory scanner + a deterministic labelling/splitting policy + a shared text
extractor + CLI wiring — no new domain, application, or ML code.

Canonical layout: `documents/<author>/<work>.txt` under a git-ignored `./documents/`
(mirroring `artifacts/`). The parent directory is the author (label); the filename is
the work (provenance). Authors are sorted lexicographically and given contiguous
integer labels for cross-machine determinism. Each work is stripped of Project
Gutenberg boilerplate and split into paragraphs (≥ a minimum length); paragraphs are
the `EvaluationSample`s.

Three decisions keep it honest and scalable:

1. **Reuse the port, don't fork it.** Satisfying `DatasetRepository`
   (`get_samples`/`get_label_names`/`get_num_classes`) means zero changes upstream;
   `seeded_subsample` already gives stratified, deterministic budgeting.
2. **Work-level holdout by default.** For authors with ≥2 works, whole works go to
   train xor test so the authorship-accuracy axis measures generalisation, not
   same-work memorisation; paragraph-level split is an explicit opt-in.
3. **One shared extractor.** Gutenberg-strip + paragraph-split + path-parse move from
   `visualize_corpus` into `shared/document_corpus.py`; the adapter and the CLI call
   the same pure functions (behaviour preserved, deduplicated).

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- No changes. Reuse `EvaluationSample` and the `DatasetRepository` ABC. (Add an
  architecture test asserting the new adapter is a `DatasetRepository` and that the
  shared corpus module stays free of upper-layer imports.)

### Application Layer (`src/colors_of_meaning/application/`)

- No changes. Verified by an integration test that runs `EvaluateUseCase` /
  `RateDistortionSweepUseCase` against the new adapter over a synthetic tree.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/dataset/document_corpus_dataset_adapter.py`:
  `DocumentCorpusDatasetAdapter` implementing `DatasetRepository`. Responsibilities:
  `pathlib` scan of `documents/<author>/<work>.txt`; parse author/work via the shared
  module; read each work (`utf-8`, `errors="ignore"`); extract paragraphs (shared);
  cap with `paragraphs_per_work`; assign sorted author labels; build the deterministic
  per-author train/test split (work-level default, paragraph fallback); skip
  non-`.txt`, empty, or too-few-paragraph cases; apply `seeded_subsample` for
  `max_samples`. Caches the scan so `train`/`test` calls are consistent.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/rate_distortion.py`: add `--source {dataset,documents}`,
  `--documents-dir`, `--min-paragraph-chars`, `--paragraphs-per-work`,
  `--split-strategy {work,paragraph}`; `_setup_dataset` branches to build the corpus
  adapter when `source == documents`.
- `interface/cli/visualize_corpus.py`: replace its private
  `_strip_gutenberg_boilerplate`/`_extract_paragraphs` with the shared functions
  (no behaviour change).
- `.gitignore`: add `documents/`.
- `README.MD`: document the `documents/<author>/<work>.txt` convention and a
  `tox -e rate_distortion -- --source documents --documents-dir ./documents` example.
- Architecture test: the corpus adapter imports stay within infrastructure rules.

### Shared Layer (`src/colors_of_meaning/shared/`)

- `shared/document_corpus.py` (pure, no I/O): `strip_gutenberg_boilerplate(text)`,
  `extract_paragraphs(text, min_chars) -> List[str]`, `parse_author_work(path) ->
  Tuple[str, str]`.

## Dependency Injection

The CLI constructs `DocumentCorpusDatasetAdapter` from the args and injects it as the
`DatasetRepository`, exactly where the HF adapters are built today; the rest of the
graph (encoder, classifier, evaluate/sweep use cases) is unchanged. No Lagom/API
changes.

## Task List

1. [ ] shared: `document_corpus.py` (`strip_gutenberg_boilerplate`,
   `extract_paragraphs`, `parse_author_work`) + pure unit tests.
2. [ ] infrastructure: `DocumentCorpusDatasetAdapter` + tests over a synthetic
   `tmp_path` tree (sorted labels; multi-work-per-author aggregation; deterministic
   work-level split with no leakage; paragraph fallback; `max_samples` stratified;
   skip non-`.txt`/empty/too-few; repeated-load identity).
3. [ ] interface: `rate_distortion` `--source documents` wiring + tests (mock the
   adapter/sweep); refactor `visualize_corpus` to the shared extractor (existing tests
   stay green); `.gitignore` `documents/`; README.
4. [ ] architecture tests: adapter is a `DatasetRepository`; shared module import
   hygiene; corpus adapter independence from application/interface.
5. [ ] integration (marked): build a synthetic 2-author tree, run
   `RateDistortionSweepUseCase` (distortion) and `EvaluateUseCase` (author accuracy)
   end-to-end; assert deterministic repeated runs.
6. [ ] migration (local, optional): move the four `reports/*.txt` books to
   `documents/<author>/<work>.txt`; produce a local documents-frontier (not committed
   to CI, since `./documents/` is git-ignored).
7. [ ] run `tox`; confirm 8 gates + 100% coverage.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names, no
network, hermetic filesystem via `tmp_path`. Key tests:

- **Shared extractor:** boilerplate stripped between Gutenberg markers; paragraphs
  below `min_chars` dropped; `parse_author_work` returns (parent-dir, filename-stem).
- **Adapter (synthetic tree):** author labels are sorted/contiguous; multiple works
  per author share a label; `work` split is disjoint with no test-work paragraph in
  train; `paragraph` fallback for a single-work author; `max_samples` is stratified
  and reproducible; non-`.txt`/empty/too-few-paragraph inputs are skipped; two loads
  with the same seed are identical.
- **CLI:** `--source documents` builds the corpus adapter with the parsed args (mock
  the adapter + sweep); `--source dataset` path unchanged.
- **`visualize_corpus`:** unchanged outputs after the shared-extractor refactor.
- **Integration (marked):** author-accuracy is measurable and repeated runs match.
- **Committed frontier:** produced locally from real books, not asserted in CI tests.

## Observability Plan

`correlation-id` logging at discovery: `{authors, works_per_author, total_paragraphs,
train_size, test_size, split_strategy}`. No new metrics/tracing.

## Risks and Mitigations

- **Train/test leakage** (same work in both splits inflates authorship accuracy).
  Mitigation: work-level holdout by default; a test asserts no test-work paragraph
  appears in train.
- **Scale/cost** (embedding many paragraphs; EMD k-NN at 4096 bins). Mitigation:
  `paragraphs_per_work` + `min_paragraph_chars` caps, lexicographic ordering, and
  `seeded_subsample` budgeting; the accuracy axis stays bounded by `max_samples` and a
  cheap distance (JS/sliced, feature 019).
- **Git-ignored corpus absent in CI.** Mitigation: every test builds a synthetic
  `tmp_path` tree; no committed artifact depends on `./documents/`; the 100%-coverage
  gate never requires real documents.
- **Cross-machine non-determinism.** Mitigation: sort authors and works before
  labelling/splitting; seed every RNG; a repeated-load test asserts identical output.
- **Refactor regression** moving extraction out of `visualize_corpus`. Mitigation:
  behaviour-preserving move; the existing `visualize_corpus` tests must stay green.
- **Scope creep** into new CLIs/datasets. Mitigation: wire only `rate_distortion`
  now; `eval`/`eval_suite` adoption is a one-line follow-up since the adapter is a
  plain `DatasetRepository`.
