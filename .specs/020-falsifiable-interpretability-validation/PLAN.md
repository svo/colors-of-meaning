# Plan: Falsifiable interpretability validation of the structured color mapper

## Implementation Strategy

Turn the "hue=topic, lightness=sentiment, chroma=concreteness" claim from
true-by-construction into measured-and-falsifiable by adding an interpretability
evaluator behind a domain port, scoring the three axes on a **held-out** split for
the **structured** mapper, and proving the scores collapse for a **negative
control** (the unconstrained mapper and/or an untrained noise projector).

The structured mapper is trained toward these axes (`_derive_hue_targets`,
`_derive_lightness_targets`, `_derive_chroma_targets`
`structured_pytorch_color_mapper.py:251–306`), so a high held-out score is only
meaningful relative to a control that was never trained toward them. The headline
result is the **margin** structured − control per axis; if any margin is
non-positive, the run reports that axis as **falsified** rather than passing.

Three decisions keep it scientific and within the layer rules:

1. **Margin, not magnitude.** Interpretability is asserted only when the
   structured mapper beats the control by a documented margin on an axis. This is
   the falsifiability gate and the project's "science not plumbing" stance.
2. **Metric self-validation.** A synthetic test constructs colors that encode a
   known axis and checks the evaluator recovers a high score there and a low score
   on an unrelated axis — so a passing real number is trustworthy.
3. **Keep numerics in infrastructure.** `scikit-learn`/`scipy` live in the
   evaluator adapter behind `InterpretabilityEvaluator`, mirroring
   `SpearmanStructurePreservationEvaluator`; the domain stays pure.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/service/interpretability_evaluator.py` → `InterpretabilityEvaluator(ABC)`
  with `evaluate(lab_colors, topics, sentiments, concreteness) ->
  InterpretabilityReport`.
- `domain/model/interpretability_report.py` → frozen `InterpretabilityReport`
  with the structured and control scores per axis, computed `margins`, a
  `falsified_axes: List[str]`, and an `is_validated` property. Validates score
  ranges (NMI in `[0,1]`, correlations in `[-1,1]`). No `sklearn`/`scipy` import.
- Reuses `ConcretenessLexicon`, `LabColor`.
- Tests: report validation, `margins`, `falsified_axes`/`is_validated` boundary.

### Application Layer (`src/colors_of_meaning/application/`)

- `EvaluateInterpretabilityUseCase(embedding_adapter, structured_mapper,
  control_mapper, evaluator, concreteness_lexicon)` → `execute(samples) ->
  InterpretabilityReport`: embed held-out docs once; map to Lab with both mappers;
  collect topic labels, the sentiment signal, and per-doc concreteness; score both
  mappers via the evaluator; assemble the report. `correlation-id` logging.
- Depends only on injected ports.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/evaluation/sklearn_interpretability_evaluator.py` →
  `SklearnInterpretabilityEvaluator(InterpretabilityEvaluator)`:
  - hue↔topic: hue angle `atan2(b*, a*)` of each mean Lab color → `num_bins`
    arcs → `normalized_mutual_info_score`/`adjusted_rand_score` vs gold topics.
  - lightness↔sentiment: `scipy.stats.pointbiserialr` (binary) or `spearmanr`
    (graded) of mean `L*` vs the sentiment signal.
  - chroma↔concreteness: `spearmanr` of mean chroma `sqrt(a*²+b*²)` vs Brysbaert
    scores.
- Reuses `BrysbaertConcretenessLexicon`. No change to existing evaluators.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/interpretability.py` (tyro `@dataclass`, `main(args)`,
  `__main__`): build structured + control mappers via `create_color_mapper`, the
  embedding adapter, the lexicon, and the evaluator; run the use case; print the
  per-axis structured/control/margin table and the verdict; write
  `reports/interpretability.md`.
- `configs/interpretability.yaml` (dataset, model paths, per-axis margins).
- `tox.ini`: `[testenv:interpretability]`.
- Architecture test: add `interpretability` to the CLI→use-case rule; assert the
  domain `interpretability_evaluator`/`interpretability_report` do **not** import
  `sklearn`/`scipy`, and that the sklearn evaluator may.
- `README.MD`: "Interpretability (Validated)" subsection (by-construction vs
  measured + control).

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

The CLI constructs the two mappers, embedding adapter, concreteness lexicon, and
the sklearn evaluator and injects them into the use case, as `train.py`/`eval.py`
already construct mappers via the factory. No Lagom/API changes.

## Task List

1. [ ] domain: `InterpretabilityReport` model + `InterpretabilityEvaluator` port +
   tests (range validation, margin computation, falsified/validated boundary,
   ABC no-instantiate/concrete-subclass).
2. [ ] infrastructure: `SklearnInterpretabilityEvaluator` + tests, including the
   **metric self-validation** (synthetic colors that encode a known axis → high on
   that axis, low on an unrelated one) and a negative case (random colors → low on
   all axes).
3. [ ] application: `EvaluateInterpretabilityUseCase` + tests (mock the mappers,
   embedding adapter, evaluator; assert structured and control are both scored and
   the report carries margins).
4. [ ] interface: `interpretability` CLI + config + tox env + architecture-test
   wiring + README subsection (mock use case for the CLI test).
5. [ ] integration (marked): train/load a structured mapper + control, run on a
   held-out split, confirm the structured margin is positive on the axes, commit
   `reports/interpretability.md`.
6. [ ] run `tox`; confirm 8 gates + 100% coverage.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names, no
network (the Brysbaert lexicon is bundled; mock the embedding adapter and dataset).
Key tests:

- **Metric self-validation:** synthetic Lab colors whose hue is set from a known
  cluster → high hue↔topic NMI; lightness set from a known binary signal → high
  lightness↔sentiment correlation; chroma set from known concreteness → high
  chroma↔concreteness; an unrelated axis on the same data → low. Random colors →
  low on all three (the null).
- **Falsifiability:** a report where the control matches the structured score
  yields a non-positive margin and the axis appears in `falsified_axes`.
- **Determinism:** two evaluations of the same colors/signals → identical scores.
- **Report model:** range validation rejects out-of-range scores; `is_validated`
  flips exactly when all margins clear their thresholds.
- **Use-case/CLI branches:** mock collaborators; cover structured-vs-control
  scoring, the sentiment-signal source selection, and stdout vs report-file write.
- **Real held-out margins:** produced by the integration run and committed to
  `reports/interpretability.md`, not asserted in unit tests.

## Observability Plan

`correlation-id` logging in the use case: the three structured scores, the three
control scores, and `falsified_axes`. Mirrors the structure-preservation
evaluator's logging. No new metrics/tracing.

## Risks and Mitigations

- **Claim true only by construction.** Mitigation: held-out split + mandatory
  negative control; report the margin, flag non-positive margins as falsified.
- **Metric measures an artifact.** Mitigation: the synthetic self-validation test
  proves the metric responds to the intended axis and not to unrelated structure.
- **Sentiment signal unavailable for a topic dataset.** Mitigation: default to the
  dataset's sentiment label where it is sentiment (IMDB); otherwise a bundled
  lexicon; never a network call.
- **Layer leakage.** Mitigation: `sklearn`/`scipy` confined to the infrastructure
  evaluator behind the port; architecture test forbids them in the domain.
- **Hue wrap-around / binning sensitivity.** Mitigation: circular hue binning into
  the mapper's cluster count; determinism test pins the result; binning sweep
  deferred (Open Questions).
- **Overclaiming significance.** Mitigation: report margins now; a permutation-test
  p-value is noted as future hardening rather than implied.
