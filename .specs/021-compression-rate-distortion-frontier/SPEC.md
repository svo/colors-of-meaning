# Feature: Rate–distortion frontier for semantic color compression

## Overview

The project's "extreme compression" headline — an exact ~1024:1 ratio (a 12,288-bit
embedding to a 12-bit color code) — is a **single operating point**, not a curve.
The honest scientific artifact for a compressor is a **rate–distortion frontier**:
how distortion changes as you spend more or fewer bits, plotted against the
alternative codecs at the **same** bit budgets. The pieces already exist —
`CompressionComparisonUseCase`
(`src/colors_of_meaning/application/use_case/compression_comparison_use_case.py`)
runs `CompressionBaseline`s (color-VQ, gzip, Product Quantization) and each returns
a `CompressedResult` with `compressed_size_bits`, `compression_ratio`, and a
`reconstruction_error` (mean ΔE distortion for the color grid) — but they are only
ever evaluated at one budget, so there is no frontier and no fair matched-budget
comparison.

This feature sweeps the **bit budget** and produces the frontier. For the color
codebook the budget knob is the grid resolution (`bins_per_dimension` → bits =
`log2(num_bins)`); for Product Quantization it is the number of subquantizers /
bits per subquantizer; gzip is a single fixed point. At each budget it records two
distortion axes: the intrinsic **ΔE reconstruction error** (already produced by the
baselines) and the **downstream consequence** — the color method's retrieval
accuracy when documents are quantized at that budget (reusing the evaluation
pipeline). The result is a committed table and a figure showing where color-VQ
sits relative to gzip and PQ on the rate–distortion (and rate–accuracy) plane.

This turns "1024:1" from a slogan into a **Pareto frontier**: it shows the actual
cost of compression in both perceptual distortion and task accuracy, and whether
the color codec is competitive with, dominated by, or complementary to standard
baselines at matched bit budgets. No new third-party dependency is introduced —
the baselines, `matplotlib` renderer (`FigureRenderer`
`domain/service/figure_renderer.py`), and evaluation pipeline already exist.

## Core Domain Concepts

- **Bit budget**: the bits-per-code a codec is allowed. For color-VQ,
  `log2(num_bins)` from `bins_per_dimension`; for PQ, `subquantizers ×
  bits_per_subquantizer`; gzip is data-dependent and contributes one point.
- **Rate–distortion point**: a `(bits_per_token, reconstruction_error)` pair for a
  codec at a budget, plus an optional downstream accuracy. The atomic sample of
  the frontier.
- **Frontier / Pareto set**: the ordered set of points per codec across budgets;
  the lower-left envelope (less distortion at fewer bits) is the Pareto frontier.
- **Matched-budget comparison**: codecs compared at the **same** bits-per-token, so
  "color-VQ vs PQ" is apples-to-apples rather than at incidental operating points.
- **Downstream distortion**: the retrieval/classification accuracy drop caused by
  quantizing at a budget — distortion that matters for the task, not just ΔE.

## User Stories

- As a researcher, I want a rate–distortion **curve** for the color codec, not a
  single ratio, so the compression claim is a frontier with a measured cost.
- As a researcher, I want color-VQ compared to gzip and PQ at **matched bit
  budgets**, so the comparison is fair.
- As a researcher, I want the **task** cost of compression (accuracy vs bits), not
  only ΔE, so I can judge whether the squeeze is worth it.
- As a maintainer, I want a committed `reports/rate_distortion.md` and a figure
  regenerable from one command.
- As a contributor, I want the sweep to reuse the existing baselines, renderer, and
  evaluation rather than reimplementing compression.

## Acceptance Criteria

- [ ] Given the color codec, when the budget is swept over at least four grid
  resolutions (e.g. `bins_per_dimension` ∈ {2, 4, 8, 16} → 3/6/9/12 bits), then a
  `(bits_per_token, reconstruction_error)` point is recorded for each.
- [ ] Given PQ and given gzip, when evaluated, then PQ contributes a curve across
  its budgets and gzip contributes its single point, all on the same axes as
  color-VQ.
- [ ] Given a target bit budget present for two codecs, when compared, then the
  matched-budget distortions are reported side by side.
- [ ] Given the color codec at each budget, when the downstream retrieval accuracy
  is measured on a fixed sample, then a `(bits_per_token, accuracy)` point is
  recorded, yielding a rate–accuracy curve alongside rate–distortion.
- [ ] Given the sweep, when a figure is rendered, then it shows rate vs distortion
  (and rate vs accuracy) at **exact, non-cropped** axes and is written to
  `reports/figures/rate_distortion.png`.
- [ ] Given the same seed and budgets, when the sweep runs twice, then the recorded
  numbers are identical.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays 100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No new business rules. A small immutable `domain/model/rate_distortion_point.py`
→ frozen `RateDistortionPoint(method, bits_per_token, reconstruction_error,
accuracy: Optional[float])` represents one sample; a `RateDistortionFrontier`
value object can order points and expose the Pareto envelope. Reuses
`CompressionBaseline`, `CompressedResult`, `FigureRenderer`. No domain service
depends on a framework.

### Application Layer (`src/colors_of_meaning/application/`)

New `RateDistortionSweepUseCase(baseline_factory, evaluate_use_case_factory)` →
`execute(embeddings, budgets) -> RateDistortionFrontier`: for each budget, build
the codec at that budget, compress to get `(bits, ΔE)`, optionally run the
downstream evaluation at that budget for `(bits, accuracy)`, and collect the
points. Reuses `CompressionComparisonUseCase` for the per-budget compression and
`EvaluateUseCase` for the downstream axis. `correlation-id` logging.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new compression code — the existing `ColorVqCompressionBaseline`,
`GzipCompressionBaseline`, and `PqCompressionBaseline` are parameterized per
budget. The `MatplotlibFigureRenderer` gains a `render_rate_distortion(frontier,
output_path)` method (fixed-axis plot, markers per codec, log-or-linear bits axis),
behind a new `FigureRenderer` method so the domain port stays the boundary.

### Interface Layer (`src/colors_of_meaning/interface/`)

New `interface/cli/rate_distortion.py` (tyro `@dataclass`, `main(args)`,
`__main__`): loads embeddings (from a trained encode run or a config), runs the
sweep over a configurable budget list, prints the matched-budget table, writes
`reports/rate_distortion.md`, and renders `reports/figures/rate_distortion.png`.
New `[testenv:rate_distortion]`. README "Extreme Compression" / "Compression
Baselines" section gains the frontier figure and a pointer to the committed table.
No API endpoint.

### Shared Layer

No changes.

## API Contracts

No API contract changes. The frontier is an offline/CLI research artifact; the
`POST /query/palette` contract is unaffected.

## CLI Impact

One new CLI, `rate_distortion`: `--config`, `--budgets` (e.g. `2,4,8,16` grid
resolutions), `--methods` (`color_vq,gzip,pq`), `--with-accuracy` (run the
downstream axis), `--max-samples`, `--output-path` (default
`reports/rate_distortion.md`), `--figure-path` (default
`reports/figures/rate_distortion.png`). No existing CLI changes.

## Dependency Injection

The CLI constructs the baselines per budget, the figure renderer, and (optionally)
the evaluation pipeline, injecting them into the sweep use case — matching the
`compress`/`eval` construction style. No Lagom/API changes.

## Observability

`correlation-id` logging in the sweep: per budget `{method, bits_per_token,
reconstruction_error, accuracy}` and a summary of the Pareto envelope. No new
metrics/tracing.

## Open Questions

- **Downstream axis cost.** Running the full evaluation at every budget is
  expensive. Default: a fixed modest sample for the accuracy axis, recorded as the
  budget; full-set is opt-in (and benefits from feature 019's fast distance).
- **PQ budget parameterization.** Default: sweep subquantizers at fixed bits each;
  exposing bits-per-subquantizer is a future option.
- **Distortion metric for non-color codecs.** gzip/PQ reconstruction error is
  defined in embedding space, color-VQ in ΔE; the figure plots each codec's native
  distortion and the matched-budget table notes the metric per row.
- **Theoretical bound.** Overlaying an entropy/rate-distortion lower bound is a
  future enhancement.
