# Plan: Rate–distortion frontier for semantic color compression

## Implementation Strategy

Replace the single "~1024:1" operating point with a measured **rate–distortion
frontier**: sweep the bit budget for each codec, record `(bits_per_token,
ΔE-distortion)` and a downstream `(bits_per_token, accuracy)` point, and plot
color-VQ against gzip and PQ at matched budgets. All compression already exists —
`CompressionComparisonUseCase` runs `CompressionBaseline`s that each return a
`CompressedResult` with bits and `reconstruction_error` — so the work is a sweep
orchestrator, a small frontier value object, a renderer method, and a CLI.

The budget knob differs per codec: color-VQ uses the codebook resolution
(`bins_per_dimension` → `log2(num_bins)` bits), PQ uses subquantizers × bits, and
gzip is a single data-dependent point. The sweep builds each codec at each budget,
compresses the same embeddings, and (optionally) runs the existing evaluation at
that budget for the task-cost axis.

Three decisions keep it honest and cheap:

1. **Reuse, don't reimplement.** The baselines, `FigureRenderer`, and
   `EvaluateUseCase` are reused; the sweep only parameterizes and collects.
2. **Two distortion axes.** Intrinsic ΔE *and* downstream accuracy, because a
   compressor that preserves ΔE but wrecks retrieval is not actually good.
3. **Matched budgets.** Points are compared at equal bits-per-token so the
   color-VQ-vs-PQ claim is apples-to-apples, not at incidental operating points.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/model/rate_distortion_point.py` → frozen `RateDistortionPoint(method,
  bits_per_token, reconstruction_error, accuracy: Optional[float])` and
  `RateDistortionFrontier(points)` with `pareto_envelope()` and `at_budget(bits)`
  for matched comparison. Pure; validates non-negative bits/error.
- `domain/service/figure_renderer.py`: add `render_rate_distortion(frontier,
  output_path) -> None` to the `FigureRenderer` ABC.
- Tests: point validation, Pareto envelope selection, matched-budget lookup.

### Application Layer (`src/colors_of_meaning/application/`)

- `RateDistortionSweepUseCase(baseline_factory, evaluate_use_case_factory=None)`
  → `execute(embeddings, budgets, methods, with_accuracy) ->
  RateDistortionFrontier`: for each (method, budget), build the codec, compress to
  `(bits, ΔE)`, optionally evaluate to `(bits, accuracy)`, collect points.
  `correlation-id` logging.
- Reuses `CompressionComparisonUseCase` for per-budget compression and
  `EvaluateUseCase` for the accuracy axis.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `MatplotlibFigureRenderer`: implement `render_rate_distortion` — fixed-axis
  (no `bbox_inches="tight"` crop that would distort the plot), one series per
  codec, markers + legend, distortion on the left axis and (if present) accuracy on
  a twin axis. Save via the existing `_save_figure` discipline.
- No new compression code; `ColorVqCompressionBaseline`/`PqCompressionBaseline`
  are constructed per budget by the CLI/factory.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/rate_distortion.py` (tyro `@dataclass`, `main(args)`,
  `__main__`): resolve embeddings, build the baseline factory and figure renderer,
  run the sweep, print the matched-budget table, write
  `reports/rate_distortion.md`, render `reports/figures/rate_distortion.png`.
- `tox.ini`: `[testenv:rate_distortion]`.
- Architecture test: add `rate_distortion` to the CLI→use-case rule.
- `README.MD`: add the frontier figure + table pointer to the compression section.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

The CLI constructs per-budget baselines, the figure renderer, and optionally the
evaluation pipeline, injecting them into the sweep use case, matching the
`compress`/`eval` construction style. No Lagom/API changes.

## Task List

1. [ ] domain: `RateDistortionPoint` + `RateDistortionFrontier` (+ `pareto_envelope`,
   `at_budget`) + `FigureRenderer.render_rate_distortion` signature + tests.
2. [ ] application: `RateDistortionSweepUseCase` + tests (mock baselines + eval;
   assert one point per budget, accuracy axis populated only when requested,
   methods iterated).
3. [ ] infrastructure: `MatplotlibFigureRenderer.render_rate_distortion` + a real
   `tmp_path` test (file written, non-zero size, fixed axes — never mock
   matplotlib).
4. [ ] interface: `rate_distortion` CLI + tox env + architecture-test wiring +
   README (mock the sweep use case for the CLI test; real renderer in its own test).
5. [ ] integration (marked): sweep color-VQ over {2,4,8,16} + gzip + PQ on a real
   encode run; commit `reports/rate_distortion.md` and
   `reports/figures/rate_distortion.png`.
6. [ ] run `tox`; confirm 8 gates + 100% coverage.

## Testing Strategy

House rules: one logical assertion per test, `test_should_..._when_...` names, no
network (synthetic embeddings; mock the dataset/evaluation). Key tests:

- **Frontier model:** rejects negative bits/error; `pareto_envelope` keeps only
  non-dominated points; `at_budget` returns the matched points for two methods.
- **Sweep:** with mocked baselines returning known `(bits, ΔE)`, one point per
  budget is produced; `with_accuracy` adds the accuracy axis via a mocked
  `EvaluateUseCase`; multiple methods are all swept.
- **Renderer (real matplotlib):** `render_rate_distortion` writes a non-empty PNG
  at the requested path with the expected figure size — exercised via `tmp_path`,
  not a mock (017's lesson: never mock the imaging library).
- **CLI branches:** mock the sweep use case; cover budget parsing, method
  selection, the `--with-accuracy` toggle, and table/figure output paths.
- **Committed frontier:** produced by the integration run, written to
  `reports/rate_distortion.md` + the figure, not asserted in unit tests.

## Observability Plan

`correlation-id` logging in the sweep: per `(method, budget)` →
`{bits_per_token, reconstruction_error, accuracy}` and a Pareto-envelope summary.
No new metrics/tracing.

## Risks and Mitigations

- **Downstream axis is expensive.** Mitigation: a fixed modest sample for accuracy,
  recorded as the budget; full-set opt-in (and faster once feature 019 lands).
- **Cross-codec distortion metrics differ** (ΔE for color, embedding error for
  gzip/PQ). Mitigation: plot each codec's native distortion and label the metric
  per row in the matched-budget table; do not pretend they are the same number.
- **Cropped/auto-scaled figure misleads.** Mitigation: fixed axes, explicit
  limits, no `bbox_inches="tight"`; a real `tmp_path` render test guards it.
- **Sweep non-determinism.** Mitigation: fixed seed and budget list; a repeat-run
  test asserts identical recorded numbers.
- **Scope creep into new compressors.** Mitigation: reuse the three existing
  baselines only; new codecs and theoretical bounds are deferred (Open Questions).
