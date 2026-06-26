# Feature: Authored document corpus source from the filesystem

## Overview

Today the only text the pipeline can ingest comes from three hard-wired, labelled
classification datasets (`--dataset ag_news|imdb|newsgroups`), loaded over the
`datasets` library. The four real-prose works the project actually reasons about —
Austen, Darwin, Doyle, Smith — live as loose `reports/*.txt` files that **only**
`interface/cli/visualize_corpus.py` can read, and that path is not a
`DatasetRepository`, so those documents can never drive the evaluation pipeline,
the ablations, or the new rate–distortion frontier (feature 021).

This feature makes **a directory of named source documents a first-class text
source**. It introduces a canonical, discoverable home — `./documents/`
(git-ignored, like `artifacts/`) — laid out as
`documents/<author>/<work>.txt`, and a `DocumentCorpusDatasetAdapter` that
**implements the existing `DatasetRepository` port**
(`domain/repository/dataset_repository.py`). Because it satisfies the same
contract as the HF adapters, it drops without change into `EvaluateUseCase`,
`RateDistortionSweepUseCase`, `eval`, `ablate`, `eval_suite`, and
`visualize_corpus`: the **author becomes the class label**, each work contributes
paragraph-level samples, and the existing seeded/stratified sampling
(`infrastructure/dataset/seeded_sampler.py`) bounds the budget.

The design anticipates growth: many more documents, and **many works per author**.
Discovery is by convention (parent directory = author, filename = work), ordering
is lexicographic for cross-machine determinism, and the train/test split defaults
to **holding out whole works** per author so the downstream authorship-accuracy
axis measures generalization rather than memorising one work. The pure text
machinery (Gutenberg-boilerplate stripping, paragraph extraction, path parsing) is
lifted out of the `visualize_corpus` CLI into `shared/document_corpus.py` so both
the adapter and the CLI share one implementation. No new third-party dependency is
introduced.

## Core Domain Concepts

- **Source document**: one `documents/<author>/<work>.txt` file of real prose; its
  immediate parent directory names the **author** (the class label), its filename
  names the **work** (provenance).
- **Author label**: the classification target. Authors are sorted lexicographically
  and assigned contiguous integer labels so labels are stable across machines.
- **Paragraph sample**: a paragraph (blank-line-delimited block ≥ a minimum length,
  after boilerplate stripping) becomes one `EvaluationSample(text, label=author,
  split)`. Works are sampled at paragraph granularity to yield volume per class.
- **Work-level holdout**: for an author with ≥2 works, whole works are partitioned
  into train vs test so no paragraph of a test work is seen in training — the
  honest authorship-generalisation split (vs. leakage-prone paragraph-level split).
- **Document corpus directory**: the git-ignored `./documents/` root scanned at
  runtime; absent in CI, so all tests use a synthetic temporary tree.

## User Stories

- As a researcher, I want to point the pipeline at a directory of my own documents
  (named by author/work) and have author become the classification label, so I can
  run the colour method, ablations, and the rate–distortion frontier on real prose.
- As a researcher, I want **many works per author** to all count toward that
  author's class, so per-author colour distributions are rich, not single-work.
- As a researcher, I want the train/test split to hold out whole works by default,
  so the authorship-accuracy axis is not inflated by same-work leakage.
- As a maintainer, I want documents under a discoverable, git-ignored `./documents/`
  convention (not loose `reports/*.txt`), and one shared paragraph/parse
  implementation rather than logic trapped in a CLI.
- As a contributor, I want the corpus source to satisfy the existing
  `DatasetRepository` contract so every consumer works unchanged.

## Acceptance Criteria

- [ ] Given `documents/<author>/<work>.txt` files for ≥2 authors, when the adapter
  loads them, then every qualifying paragraph becomes an `EvaluationSample` whose
  `label` is the author and `get_label_names()` returns the authors in sorted order
  with contiguous integer labels.
- [ ] Given an author with multiple works, when the corpus is loaded, then all of
  that author's works contribute samples under the same author label.
- [ ] Given a fixed seed, when `get_samples("train")` and `get_samples("test")` are
  called, then the per-author split is deterministic, disjoint, and identical across
  repeated runs; with the default `work` strategy and an author of ≥2 works, no
  test-work paragraph appears in train.
- [ ] Given a `max_samples` budget, when sampling, then the result is stratified by
  author and reproducible (reusing `seeded_subsample`).
- [ ] Given the document corpus wired into the `rate_distortion` CLI with ≥2
  authors, when it runs, then the distortion frontier is produced over document
  embeddings and the rate–accuracy axis classifies documents by author.
- [ ] Given `./documents/` is git-ignored, when the test suite runs, then every test
  builds a synthetic `tmp_path` document tree (no real-file or network dependency)
  and `tox` stays green at 100% coverage.
- [ ] Given a directory containing non-`.txt` files, empty files, or an author whose
  works yield too few paragraphs to split, when loaded, then those are skipped per
  the documented rules without raising.
- [ ] Given the same seed and directory, when the corpus is loaded twice, then the
  labels, splits, and sample order are identical.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No new business rules and no new model. The adapter reuses `EvaluationSample`
(`domain/model/evaluation_sample.py`) and satisfies the existing
`DatasetRepository` interface (`domain/repository/dataset_repository.py`). Author →
integer-label mapping is data, not a domain rule.

### Application Layer (`src/colors_of_meaning/application/`)

No changes. `EvaluateUseCase`, `RateDistortionSweepUseCase`,
`AblationSweepUseCase`, and `EvaluationSuiteUseCase` already depend only on the
`DatasetRepository` abstraction and consume the document corpus unchanged.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/dataset/document_corpus_dataset_adapter.py`:
`DocumentCorpusDatasetAdapter(documents_dir, min_paragraph_chars,
paragraphs_per_work, split_strategy, test_fraction)` implementing
`DatasetRepository`. It scans `documents/<author>/<work>.txt` with `pathlib`,
delegates boilerplate-stripping/paragraph-extraction/path-parsing to
`shared/document_corpus.py`, assigns sorted author labels, builds the deterministic
train/test split (work-level by default), and reuses `seeded_subsample` for the
`max_samples` budget. Embedding stays the existing `SentenceEmbeddingAdapter`
(`encode_batch`, tunable `batch_size`/`show_progress`); no new ML code.

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/cli/rate_distortion.py` gains `--source {dataset,documents}`,
`--documents-dir`, `--min-paragraph-chars`, `--paragraphs-per-work`, and
`--split-strategy {work,paragraph}`; when `--source documents` it constructs the
`DocumentCorpusDatasetAdapter` instead of an HF adapter. `interface/cli/
visualize_corpus.py` is refactored to call the shared extractor (behaviour
preserved). `eval`/`eval_suite` may adopt the corpus via the same one-line dataset
factory addition (noted; not all wired in this feature). `.gitignore` gains
`documents/`. README documents the convention and a `tox -e rate_distortion --
--source documents` example. No API endpoint.

### Shared Layer (`src/colors_of_meaning/shared/`)

New `shared/document_corpus.py` — pure, no I/O: `strip_gutenberg_boilerplate`,
`extract_paragraphs(text, min_chars)`, and `parse_author_work(path)` →
`(author, work)`. Importable by infrastructure and interface; framework-free per the
shared-layer rules.

## API Contracts

No API contract changes. The corpus is an offline/CLI research input; the
`POST /query/palette` contract is unaffected.

## CLI Impact

`rate_distortion` gains a document-corpus source: `--source` (`dataset` default for
back-compat, or `documents`), `--documents-dir` (default `./documents`),
`--min-paragraph-chars`, `--paragraphs-per-work`, `--split-strategy`. No existing
CLI flag changes; `visualize_corpus` keeps its current flags but reads through the
shared extractor.

## Dependency Injection

The CLI selects between the HF adapters and `DocumentCorpusDatasetAdapter` by source,
constructs it from the CLI args, and injects it as the `DatasetRepository` — matching
how `eval`/`ablate` build their dataset repositories today. No Lagom/API changes.

## Observability

`correlation-id` logging at corpus discovery: authors found, works per author, total
qualifying paragraphs, per-split sizes, and the split strategy used. No new
metrics/tracing.

## Open Questions

- **Split strategy default.** Default `work`-level holdout for authors with ≥2 works
  (no same-work leakage), falling back to `paragraph` for single-work authors;
  `--split-strategy paragraph` forces the simpler leakage-prone split. Is per-author
  fallback acceptable, or should single-work authors be excluded from the accuracy
  axis entirely?
- **Sample granularity.** Default paragraph (matches `visualize_corpus`); sentence-
  or whole-work granularity is a future option.
- **Discovery metadata.** Author/work are derived from the path; an optional
  `documents/manifest.yaml` (display names, language, year) is a future enhancement.
- **Migration of `reports/*.txt`.** Move the four books to
  `documents/<author>/<work>.txt` and re-point `visualize_corpus` defaults, or keep
  `reports/*.txt` for back-compat and only add `./documents/`? Default: migrate, and
  keep `visualize_corpus`'s explicit `--corpus-specs` working against any path.
- **Committed artifact.** Because `./documents/` is git-ignored, a documents-sourced
  frontier report is local/opt-in and is **not** regenerated in CI; the committed
  AG-News frontier (021) remains the CI-reproducible one.
