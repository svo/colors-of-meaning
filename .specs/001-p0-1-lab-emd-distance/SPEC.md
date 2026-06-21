# Feature: True Lab Earth-Mover Distance (POT)

## Overview
The color-histogram distance that powers document comparison and retrieval is currently a 1-D `scipy.stats.wasserstein_distance` computed over codebook **bin indices** (`0…4095`). Index `i` and index `i+1` may be perceptually unrelated colors, so the metric measures palette ordering, not perceptual similarity, and any "color method" number derived from it is meaningless. This feature replaces that calculator with a true Earth-Mover (Wasserstein-1) distance over the **perceptual ground cost** between codebook Lab colors, using the POT (`ot`) library. The cost matrix is built once from the injected `ColorCodebook` (4096×3 Lab coordinates → 4096×4096 Euclidean cost) and reused across calls; an optional entropic-regularised Sinkhorn path is offered for corpus-scale retrieval, gated by a new `distance.sinkhorn_reg` config field. This is the first roadmap step (P0-1) and a prerequisite for an honest end-to-end AG News run (006-p0-6-end-to-end-agnews-table).

## User Stories
- As a researcher, I want document distance to reflect perceptual color geometry so that retrieval and the reported color-method accuracy measure semantic similarity rather than arbitrary palette index ordering.
- As a researcher, I want the perceptual cost matrix precomputed from the codebook and injected once so that comparing thousands of documents does not rebuild a 67 MB matrix on every call.
- As a researcher, I want an optional Sinkhorn variant gated by config so that corpus-scale retrieval can trade exactness for speed without changing call sites.
- As a maintainer, I want the distance calculator to reject documents from a different codebook so that silently comparing incompatible histograms is impossible.

## Acceptance Criteria
- [ ] Given two histograms that move mass between perceptually **close** codebook colors, when the distance is computed, then the result is **small**.
- [ ] Given two histograms that move mass between perceptually **distant** codebook colors, when the distance is computed, then the result is **large** (strictly greater than the close-colors case — the property the current index version provably fails).
- [ ] Given two identical histograms, when the distance is computed, then the result is approximately zero.
- [ ] Given a `ColorCodebook`, when the calculator is constructed, then the perceptual cost matrix is built exactly once (`ot.dist(coords, coords, metric='euclidean')`) and reused, not rebuilt per `compute_distance` call.
- [ ] Given two documents whose `num_bins` differs from the codebook size, when the distance is computed, then a `ValueError` is raised (same-codebook validation, stricter than today's `num_bins`-only check).
- [ ] Given `distance.sinkhorn_reg` is set to a positive value, when the distance is computed, then `ot.sinkhorn2(..., method='sinkhorn_log')` is used; given it is unset/`None`, then exact `ot.emd2` is used.
- [ ] Given the existing CLI commands (`eval`, `compare`, `query`, `visualize`), when they construct the calculator, then they inject the codebook and configured regularisation via their factory functions and still run.
- [ ] Given `metric_name()` is called, then it returns `"wasserstein"` (unchanged, so existing config `distance.metric == "wasserstein"` keeps selecting it).
- [ ] Given the full quality gate, when `tox` runs, then all 8 gates pass, coverage is 100%, and `pip-audit` reports no vulnerability for the new `ot` dependency.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)
No changes to the port contract. `domain/service/distance_calculator.py` (`DistanceCalculator` ABC: `compute_distance(doc1, doc2) -> float`, `metric_name() -> str`) is the interface the new implementation continues to satisfy; its signature is unchanged so domain stays pure (no `ot`/`scipy` import in domain). `domain/model/color_codebook.py` (`ColorCodebook.colors: List[LabColor]`, `num_bins`) is the read-only source of Lab coordinates for the cost matrix and is consumed, not modified. `domain/model/colored_document.py` (`histogram`, `num_bins`) is unchanged.

### Application Layer (`src/colors_of_meaning/application/`)
No changes. `application/use_case/compare_documents_use_case.py` and `application/use_case/query_by_palette_use_case.py` depend only on the `DistanceCalculator` port via constructor injection and are agnostic to the implementation swap.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- `infrastructure/ml/wasserstein_distance_calculator.py` — rewritten: constructor takes a `ColorCodebook` (and an optional `sinkhorn_reg: Optional[float]`), precomputes `coords` (4096×3 `float32`) and `M = ot.dist(coords, coords, metric='euclidean')` once at construction, and `compute_distance` returns `float(ot.emd2(doc1.histogram, doc2.histogram, M))` (or `float(ot.sinkhorn2(..., method='sinkhorn_log'))` when `sinkhorn_reg` is set). Same-codebook validation replaces the bare `num_bins` check. `scipy.stats.wasserstein_distance` import removed; `import ot` added.
- `infrastructure/evaluation/color_histogram_classifier.py` — unchanged in code; it already consumes the injected `DistanceCalculator` in `_rerank_by_distance`. This is the retrieval path that now benefits from perceptual distance.
- `infrastructure/ml/jensen_shannon_distance_calculator.py` — unchanged; documented as the **distinct** consumer of `distance.smoothing_epsilon` (1e-8), which is **not** a Sinkhorn regulariser.

### Interface Layer (`src/colors_of_meaning/interface/`)
No FastAPI controller, route, or DTO changes (the `DistanceCalculator` is not exposed over HTTP today; the query controller is unmounted dead code handled separately in 011-p2-1-mount-query-retire-coconut). CLI rewiring only, at the four arg-less construction sites:
- `interface/cli/eval.py` (`_create_color_classifier`, ~line 87) — load the codebook (already loaded a few lines above as `codebook`) and the config, pass both to the calculator.
- `interface/cli/compare.py` (~line 37) — load the codebook (new) and pass `config.distance.sinkhorn_reg`.
- `interface/cli/query.py` (~line 61) — reuse the already-loaded `codebook` and pass it in.
- `interface/cli/visualize.py` (`_create_classifier`, ~line 193) — reuse the already-loaded `codebook`.

### Shared Layer
- `shared/synesthetic_config.py` — add `sinkhorn_reg: Optional[float] = None` to `DistanceConfig` (alongside `metric`, `smoothing_epsilon`); `from_yaml`/`to_yaml` already round-trip the dataclass via `**`/`__dict__`, so they pick it up automatically.
- `shared/lab_utils.py` — unchanged; `delta_e` (CIE76 Euclidean Lab) documents the same perceptual metric the cost matrix encodes but is not imported by the calculator (the calculator vectorises via `ot.dist`).
- Config YAMLs (`configs/base.yaml`, `configs/structured.yaml`, `configs/supervised.yaml`) — optionally add an explicit `sinkhorn_reg:` under `distance:`; omission is valid (defaults to `None` → exact EMD).

## API Contracts
None. No new or modified FastAPI endpoints.

## CLI Impact
No new commands or user-facing arguments. Internal factory wiring only: the four call sites that build `WassersteinDistanceCalculator()` arg-less now build it with the loaded `ColorCodebook` and `config.distance.sinkhorn_reg`. `compare.py` gains internal codebook loading (it currently does not load one).

## Dependency Injection
- **CLI factory wiring (constructor injection):** each CLI factory function builds `WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=config.distance.sinkhorn_reg)` and injects it into `ColorHistogramClassifier` / `CompareDocumentsUseCase` as today.
- **API Lagom container:** `interface/api/main.py` does not currently register `DistanceCalculator`; no registration is added in this step (no API path uses it yet). If a later step (011-p2-1-mount-query-retire-coconut) mounts the query controller, the abstract `DistanceCalculator` should map to the concrete `WassersteinDistanceCalculator` built from the runtime codebook — noted as future, out of scope here.
- **Abstract → concrete mapping:** `DistanceCalculator` → `WassersteinDistanceCalculator(codebook, sinkhorn_reg)`.
- **New dependency:** `ot` (POT, "Python Optimal Transport") added to `setup.cfg` `install_requires` and present transitively for `pip-audit`; pin/declare in `pyproject.toml` build metadata only if required by the existing convention (the project declares runtime deps in `setup.cfg`).

## Observability
Distance computation is a hot inner loop (called per candidate pair during reranking), so per-call logging is inappropriate. Add, at construction of the calculator, a single structured `INFO`/`DEBUG` log entry with `correlation-id` recording codebook size, cost-matrix shape, and whether the exact (`emd2`) or Sinkhorn (`sinkhorn_reg`) path is active, so a run's distance configuration is auditable. No new metrics or tracing spans are mandated for the inner loop; if the project's existing observability decorators wrap use cases, the unchanged `CompareDocumentsUseCase` retains its current tracing.

## Open Questions
- **Cost-matrix dtype/precision:** `float32` keeps the 4096×4096 matrix at ~67 MB but `ot.emd2`'s LP solver may upcast or warn; confirm whether `float64` (~134 MB) is needed for solver stability/accuracy, or whether normalising `M` (e.g. `M /= M.max()`) to bound magnitudes is preferred. (Affects the "large vs small" assertion thresholds.)
- **Same-codebook validation strategy:** the simplest robust check is `doc.num_bins == codebook.num_bins` for both documents (cheap, catches the real failure mode). A stronger identity check (same codebook object/hash) is not currently expressible because `ColoredDocument` does not carry a codebook reference. Confirm `num_bins == codebook.num_bins` is sufficient, or whether `ColoredDocument` should later gain a codebook identifier (would be a domain-model change, out of scope here).
- **Default `sinkhorn_reg` for retrieval:** exact `emd2` over 4096 bins per candidate pair may be too slow for corpus-scale `eval`. Should `eval`/`compare` default to a documented small `sinkhorn_reg` (e.g. `0.01` on a normalised `M`), or stay exact by default and let the operator opt into Sinkhorn? Trade-off between fidelity and runtime for 006-p0-6-end-to-end-agnews-table.
- **POT version / pip-audit:** confirm the resolved `ot` version is free of advisories at implementation time; pin a minimum if pip-audit flags an older transitive pull.
