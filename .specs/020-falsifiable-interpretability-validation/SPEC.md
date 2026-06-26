# Feature: Falsifiable interpretability validation of the structured color mapper

## Overview

The project's signature claim is that the **structured** color mapper produces
*interpretable* colors: **hue encodes semantic cluster, lightness encodes
sentiment, chroma encodes concreteness** (`StructuredPyTorchColorMapper`
`src/colors_of_meaning/infrastructure/ml/structured_pytorch_color_mapper.py:245`).
Today that claim is true **by construction** — the mapper is *trained toward*
KMeans-cluster hue targets, sentiment-derived lightness targets, and
Brysbaert-concreteness chroma targets (`_derive_hue_targets` `:251`,
`_derive_lightness_targets` `:278`, `_derive_chroma_targets` `:291`). What is
**not** established is whether those axes actually carry that meaning on
**held-out** data as a faithful, measurable correlation, or whether the network
merely memorized the training targets. An interpretability claim that is only
true-by-construction is not yet science.

This feature makes the claim **falsifiable**. It measures, on a held-out split,
three independent correlations:

- **hue ↔ topic**: cluster the documents by hue angle of their mean color and
  score agreement with the gold topic labels (normalized mutual information /
  adjusted Rand index).
- **lightness ↔ sentiment**: correlate each document's mean `L*` with a sentiment
  signal (the binary IMDB label, or a sentiment lexicon score) via point-biserial
  / Spearman.
- **chroma ↔ concreteness**: correlate each document's mean chroma
  `sqrt(a*² + b*²)` with its Brysbaert concreteness score (reusing
  `BrysbaertConcretenessLexicon`
  `src/colors_of_meaning/infrastructure/ml/brysbaert_concreteness_lexicon.py`).

Crucially, it includes a **negative control**: the same three measurements run
against the **unconstrained** mapper (and/or an untrained noise projector), which
is *not* trained toward these axes and must therefore score near zero. The test of
the mechanism is not "the structured mapper's number is high" but "the structured
mapper's number is **significantly higher than the control's**". If the control
also scores high, the metric is measuring an artifact, not interpretability — and
the feature is designed to surface exactly that.

This mirrors the existing structure-preservation pattern
(`SpearmanStructurePreservationEvaluator`
`src/colors_of_meaning/infrastructure/evaluation/structure_preservation_evaluator.py`):
a numerical evaluator behind a domain port, kept out of the domain layer because
it depends on `scikit-learn`/`scipy` (already declared in `setup.cfg`). No new
third-party dependency is introduced.

## Core Domain Concepts

- **Interpretability axis score**: a single scalar measuring how strongly one
  perceptual channel (hue / lightness / chroma) tracks an external semantic signal
  (topic / sentiment / concreteness) on held-out documents. Hue↔topic uses a
  clustering-agreement score (NMI/ARI) in `[0, 1]`; lightness↔sentiment and
  chroma↔concreteness use rank/point-biserial correlation in `[-1, 1]`.
- **Negative control**: a mapper not trained toward the axes (unconstrained or
  noise). Its scores form the null baseline; interpretability is the **margin**
  of the structured mapper over the control, not the raw score.
- **Interpretability report**: an immutable record of the three axis scores for
  the structured mapper and the control, plus the pass/fail margins — the
  committed evidence behind the "colors of meaning" claim.
- **Mean color of a document**: the chroma/lightness/hue are read from the
  document's mean Lab color (or dominant codebook color), not a single sentence,
  so the measurement is at the document granularity the claim is stated at.

## User Stories

- As a researcher, I want to know whether hue **actually** predicts topic,
  lightness **actually** tracks sentiment, and chroma **actually** tracks
  concreteness on unseen data — measured, not asserted.
- As a skeptic, I want a **negative control** that fails these tests, so a high
  score means structure and not an artifact of the metric.
- As a maintainer, I want a committed `reports/interpretability.md` with the three
  axis scores, the control scores, and the margins, regenerable from one command.
- As a contributor, I want the interpretability metrics behind a clean domain port
  so the science is unit-testable without `scikit-learn` leaking into the domain.

## Acceptance Criteria

- [ ] Given a trained **structured** mapper and a held-out split, when
  interpretability is evaluated, then hue↔topic (NMI), lightness↔sentiment
  (correlation), and chroma↔concreteness (correlation) are each reported.
- [ ] Given the **unconstrained/noise** control on the same split, when the same
  metrics are computed, then the structured mapper's score **exceeds the control's
  by a documented margin** on each axis; if any axis fails the margin, the run
  reports it as a **falsification**, not a pass.
- [ ] Given a synthetic dataset where color is constructed to encode a known axis,
  when the evaluator runs, then it recovers a high score on that axis and a low
  score on an unrelated axis (the metric itself is validated).
- [ ] Given the same mapper, split, and seed, when the evaluation runs twice, then
  the reported scores are identical.
- [ ] Given `chroma↔concreteness`, when computed, then it uses the bundled
  Brysbaert lexicon (no network), so the run is offline and deterministic.
- [ ] Given the committed `reports/interpretability.md`, when regenerated, then it
  matches the command output.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays 100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New port `domain/service/interpretability_evaluator.py` →
`InterpretabilityEvaluator(ABC)` with `evaluate(lab_colors, topics, sentiments,
concreteness) -> InterpretabilityReport`. New model
`domain/model/interpretability_report.py` → frozen `InterpretabilityReport`
carrying `hue_topic_score`, `lightness_sentiment_score`,
`chroma_concreteness_score` (and the control's three), with `margins` and a
`falsified_axes` list. Correlations may be negative, so this is **not**
`EvaluationResult` (which validates `[0, 1]`). The domain stays free of
`scikit-learn`/`scipy` (architecture test enforces it, like the existing
scipy/sklearn isolation rules). Reuses `ConcretenessLexicon`
(`domain/service/concreteness_lexicon.py`) and `LabColor`.

### Application Layer (`src/colors_of_meaning/application/`)

New `EvaluateInterpretabilityUseCase(embedding_adapter, structured_mapper,
control_mapper, interpretability_evaluator, concreteness_lexicon)` →
`execute(samples) -> InterpretabilityReport`: embed the held-out documents, map
each to Lab with both the structured mapper and the control, gather topic labels /
sentiment signal / concreteness scores, and delegate scoring to the evaluator for
each mapper. Adds `correlation-id` logging. Depends only on injected ports.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/evaluation/sklearn_interpretability_evaluator.py` →
`SklearnInterpretabilityEvaluator(InterpretabilityEvaluator)`: hue↔topic via
`sklearn.metrics.normalized_mutual_info_score` / `adjusted_rand_score` over
hue-angle bins vs gold topics; lightness↔sentiment and chroma↔concreteness via
`scipy.stats` point-biserial/Spearman. Pure numerics behind the port, exactly
where the existing Spearman evaluator lives. Reuses `BrysbaertConcretenessLexicon`.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/interpretability.py` (tyro `@dataclass`, `main(args)`,
`__main__` guard): loads the structured mapper (and a control), a dataset, runs
`EvaluateInterpretabilityUseCase`, prints the three axes + control + margins, and
writes `reports/interpretability.md`. New `[testenv:interpretability]`. New
`configs/interpretability.yaml` (dataset, mapper paths, axis margins). README
gains an "Interpretability (Validated)" subsection contrasting *by-construction*
with *measured-on-held-out + negative-control*. No API endpoint.

### Shared Layer

No changes.

## API Contracts

No API contract changes. Interpretability evaluation is an offline/CLI research
concern; the `POST /query/palette` contract is unaffected.

## CLI Impact

One new CLI, `interpretability`: `--config`, `--dataset`, `--structured-model`,
`--control-model` (or `--control noise`), `--codebook`, `--max-samples`,
`--output-path` (default `reports/interpretability.md`). Prints the per-axis
structured vs control scores and the pass/falsified verdict. No existing CLI
changes.

## Dependency Injection

The CLI constructs the structured and control mappers (via the existing
`create_color_mapper` factory), the embedding adapter, the concreteness lexicon,
and the `SklearnInterpretabilityEvaluator`, injecting them into the use case —
matching the `eval.py`/`train.py` construction style. No Lagom/API changes.

## Observability

`correlation-id` logging in the use case: `{hue_topic_score,
lightness_sentiment_score, chroma_concreteness_score, control_hue_topic, ...,
falsified_axes}`. Consistent with the structure-preservation evaluator's logging.
No new metrics/tracing.

## Open Questions

- **Sentiment signal source.** Default: the dataset's own label where it is
  sentiment (IMDB); otherwise a bundled sentiment lexicon score. A gold sentiment
  dataset (SST) is a future option.
- **Hue binning granularity.** Default: bin hue into `num_clusters` arcs matching
  the structured mapper's cluster count; sweep is a future option.
- **Document color aggregation.** Default: mean Lab over the document's sentence
  colors; dominant-codebook-color is an alternative to expose later.
- **Significance testing.** Default: report the margin over the control; a
  permutation-test p-value per axis is a future hardening.
- **Which control.** Default: the unconstrained mapper (trained, but not toward
  these axes) plus an untrained noise projector as a second floor.
