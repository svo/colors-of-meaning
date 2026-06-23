# Plan: Real Compression Comparison (Matched Rate-Distortion)

## Implementation Strategy
Make color-VQ a first-class `CompressionBaseline` alongside gzip and PQ so all three flow through one port and `CompressionComparisonUseCase` produces one comparable row per method, then strip the dishonest accounting from the existing color path. Concretely:

- **One axis, one port.** `domain/service/compression_baseline.py` (`CompressionBaseline.compress -> CompressedResult`) is already implemented by gzip and PQ. Add a color-VQ implementation so the comparison table is uniform. `gzip_compression_baseline.py` is the structural template — it already does a real round-trip (`gzip.decompress`) and reports real `compressed_size_bits`, `original_size_bits`, and `reconstruction_error`.
- **Genuine color-VQ distortion = mean ΔE.** Reconstruction error is the mean of `delta_e(original_color, codebook.get_color(quantize(original_color)))` over the colors, using `shared/lab_utils.delta_e` (CIE76) — the same perceptual metric foundation laid by `001-p0-1-lab-emd-distance`. Error is `0.0` when every original color is already a centroid, and rises as the codebook coarsens (the rate-distortion property the tests pin).
- **An artifact that actually shrinks.** Replace the pickled-full-`color_sequence` "artifact" with fixed-width codes: `compressed_size_bits = num_codes * ceil(log2(num_bins)) (+ palette overhead)`, `original_size_bits = num_colors * 3 * float_bits`. Drop the fabricated `original_bits = num_tokens*8*10` and the non-decodable variable-width per-run `log2(color)` RLE in `compress_document_use_case.py` entirely.
- **PQ out-of-sample.** Split rows into a seeded train/holdout, `fit` on train, `predict`/score on holdout, report held-out distortion; deterministic fallback for inputs too small to split.
- **PREFER EDITING.** Edit `compress_document_use_case.py`, `compression_comparison_use_case.py`, `pq_compression_baseline.py`, `compress.py`, and the existing tests. The only candidate *new* src file is `infrastructure/ml/color_vq_compression_baseline.py`; if the learned codebook from `007-p1-1-learned-vq-codebook` and the port make a clean baseline awkward, fold color-VQ into the use case instead (resolved in task order below / SPEC Open Questions).
- **Depends on `001-p0-1-lab-emd-distance` and `007-p1-1-learned-vq-codebook`.** The ΔE metric reuses 001's perceptual foundation; the palette whose dequantization error we measure is the learned `ColorCodebook` from 007 (the uniform grid still works as a labelled baseline and makes the out-of-gamut rate-distortion penalty visible).

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
- `domain/service/compression_baseline.py`: reuse `CompressionBaseline`/`CompressedResult` unchanged for the port contract. Resolve the dead `CompressedResult.bits_per_dimension` (`return 0.0`): implement it as a real bits-per-unit or delete it (SPEC Open Question) so no decorative branch blocks 100% coverage. If a port extension is required for color codecs, add only a pure abstract signature (no `numpy`/`sklearn`/`ot`/`torch` body) — confirm the architecture test still sees `domain/*` importing none of those.
- `domain/model/colored_document.py`: if the decode-path Open Question resolves to "yes", add a pure `reconstructed_colors(codebook: ColorCodebook) -> List[LabColor]` delegating to `codebook.get_color` (no I/O, no new imports beyond domain). Otherwise no edit.
- `domain/model/color_codebook.py`, `domain/model/lab_color.py`: read-only; `get_color`/`quantize` consumed unchanged.

### Application Layer (`src/colors_of_meaning/application/`)
- `application/use_case/compress_document_use_case.py`: delete `_compute_rle_bits` and the `_compute_compression_ratio` fabricated `original_bits = num_tokens*8*10`. Inject a `ColorCodebook` (and `seed` if needed) via constructor. Compute fixed-width `compressed_size_bits`, honest `original_size_bits = num_colors*3*float_bits`, real `compression_ratio`, and `reconstruction_error = mean ΔE` via `shared/lab_utils.delta_e`. Keep `execute`/`execute_batch` shapes but with honest fields; aggregate real bit totals in the batch path.
- `application/use_case/compression_comparison_use_case.py`: ensure the per-method dict reports `reconstruction_error` for every method (already does for the port) and that color-VQ appears once color-VQ is a `CompressionBaseline`. Decide the per-unit denominator (rows vs codes/colors) and document it; keep the loop low-complexity.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- `infrastructure/ml/pq_compression_baseline.py`: introduce a seeded train/holdout row split; `kmeans.fit(train_subspace)` then `kmeans.predict(holdout_subspace)`; accumulate squared error on holdout rows only; `reconstruction_error` = held-out distortion. Deterministic single-/tiny-input fallback (fit==score on the one split). Keep float32 bit accounting consistent with the other methods. Extract the split into an expressively-named private helper so `compress` stays under `xenon`/`radon` limits.
- `infrastructure/ml/gzip_compression_baseline.py`: reference template; no behavioural change (optionally ensure its MSE distortion unit is labelled distinct from ΔE in the comparison output).
- `infrastructure/ml/color_vq_compression_baseline.py` (new, *only if* the port expresses it cleanly): implement `CompressionBaseline` using the injected `ColorCodebook` and `delta_e`; expressive private names (`_dequantized_colors`, `_mean_delta_e`, `_fixed_width_code_bits`), no comments. If folded into the use case instead, this file is not created.

### Interface Layer (`src/colors_of_meaning/interface/`)
- `interface/cli/compress.py`: in `_run_baseline_comparison`, add the color-VQ baseline (built with a `ColorCodebook` from `FileColorCodebookRepository`) to the `baselines` list so the table has gzip/PQ/color-VQ rows; print `reconstruction_error` per method. In `_run_vq_analysis`, report the honest shrinking-artifact bits and ΔE, and stop presenting the pickled full `color_sequence` as the compressed artifact. No new flags unless `--seed`/codec params are surfaced (Open Question).
- Pydantic DTOs / FastAPI controllers: no changes.

### Shared Layer (`src/colors_of_meaning/shared/`)
- `shared/lab_utils.py`: `delta_e` now actually called as the color-VQ distortion metric; body unchanged.
- `shared/synesthetic_config.py`: read the existing `seed` for the PQ split if threaded; add a `compression`/codec config field only if bit-accounting parameters must be configurable (Open Question) — otherwise no change.

## Dependency Injection
- CLI factories construct `CompressDocumentUseCase(codebook=codebook, seed=seed)` and the `CompressionBaseline` list `[GzipCompressionBaseline(), PQCompressionBaseline(...), ColorVqCompressionBaseline(codebook=codebook)]`, injecting the list into `CompressionComparisonUseCase` (constructor injection, as today).
- Abstract → concrete: `CompressionBaseline` → `{Gzip, PQ, ColorVq}`; codebook loaded via `FileColorCodebookRepository` (the loader the other CLIs already use).
- API `interface/api/main.py` Lagom container: no compression registration added (no API consumer); documented as out of scope.
- Test wiring: unit tests construct codecs directly with a small synthetic `ColorCodebook` of a few `LabColor`s; `CompressionComparisonUseCase` tests keep using `Mock` baselines; CLI tests patch the imported baseline/use-case symbols and assert the codebook (and seed) are passed.

## Task List
1. [ ] domain: resolve `CompressedResult.bits_per_dimension` in `domain/service/compression_baseline.py` — implement as a real bits-per-unit or remove it; (if decode-path Open Question is "yes") add pure `ColoredDocument.reconstructed_colors(codebook)`. (Covers AC: uniform `CompressedResult` fields; supports ΔE definition.)
2. [ ] application: rewrite `application/use_case/compress_document_use_case.py` — remove fabricated `original_bits = num_tokens*8*10` and `_compute_rle_bits`; inject `ColorCodebook`; compute fixed-width `compressed_size_bits` < `original_size_bits`, honest `original_size_bits = num_colors*3*float_bits`, real ratio, and `reconstruction_error = mean ΔE` via `shared/lab_utils.delta_e`. (Covers AC: ΔE error + zero-when-centroid, coarser-codebook-higher-error, artifact shrinks, honest original size, fabricated constant/RLE removed.)
3. [ ] application: update `application/use_case/compression_comparison_use_case.py` so every method (incl. color-VQ) reports `reconstruction_error` on one comparable row; pin/ document the per-unit denominator. (Covers AC: all three methods return populated `reconstruction_error` and sizes.)
4. [ ] infrastructure: fix `infrastructure/ml/pq_compression_baseline.py` — seeded train/holdout split, `fit` on train, score on holdout, held-out `reconstruction_error`, deterministic tiny-input fallback. (Covers AC: PQ out-of-sample fit; degenerate-input fallback.)
5. [ ] infrastructure: add `infrastructure/ml/color_vq_compression_baseline.py` implementing `CompressionBaseline` over an injected `ColorCodebook` with ΔE distortion and fixed-width code accounting (or fold into the use case if the port is awkward — decided in task 2). (Covers AC: color-VQ on the same axis as gzip/PQ.)
6. [ ] interface: update `interface/cli/compress.py` — add the color-VQ baseline to the comparison list, load the codebook via `FileColorCodebookRepository`, print reconstruction error per method, stop presenting the pickled `color_sequence` as the artifact. (Covers AC: CLI reports error + shrinking bits for every method in both modes.)
7. [ ] tests: rewrite `tests/colors_of_meaning/application/use_case/test_compress_document_use_case.py` (drop the fabricated-ratio and RLE tests; add ΔE/zero/coarser-codebook/shrink/honest-original tests), extend `test_pq_compression_baseline.py` (held-out split, fallback), add `test_color_vq_compression_baseline.py`, update `test_compression_comparison_use_case.py` and `test_compress.py`; confirm `pytest-archon` boundary tests stay green. (Covers AC: 100% coverage, one-assertion tests, tox green.)

## Testing Strategy
- **One assertion per test**, ML/numerical tests may group related asserts on the same result; plain `assert`/`pytest.raises` for these ML/codec tests (matching the existing `test_pq_compression_baseline.py` / `test_gzip_compression_baseline.py` style). Any base/config-dataclass tests use `assertpy` (`assert_that`).
- **Names** follow `test_should_<behaviour>_when_<condition>`.
- **Headline color-VQ tests:**
  - `test_should_return_zero_reconstruction_error_when_every_color_is_a_codebook_centroid` — colors drawn from the codebook → mean ΔE `0.0`.
  - `test_should_increase_reconstruction_error_when_codebook_is_coarser` — same colors against few-bin vs many-bin codebooks → coarse error `>=` fine error (the rate-distortion property).
  - `test_should_produce_compressed_size_smaller_than_original_when_compressing_colors` — `compressed_size_bits < original_size_bits` for a non-trivial input (artifact shrinks).
  - `test_should_compute_original_size_from_color_triples_when_compressing` — `original_size_bits == num_colors * 3 * float_bits` (not `num_tokens*8*10`).
- **Removal/honesty tests:** drop `test_should_compute_compression_ratio`/`_handle_zero_bits_in_compression_ratio`/`_compute_rle_bits` (they assert the deleted fabricated/RLE behaviour); replace with the honest-accounting tests above.
- **PQ tests:** `test_should_fit_kmeans_on_train_split_and_score_on_holdout_when_compressing` (e.g. assert error reflects held-out rows / patch to confirm `fit` and `predict` see disjoint slices); `test_should_fall_back_deterministically_when_input_too_small_to_split`. Keep `test_should_compute_correct_original_size` and `test_should_return_correct_name`.
- **Comparison-use-case tests:** extend the `Mock`-baseline tests to assert a color-VQ row carries a populated `reconstruction_error`; keep ratio/bits-per-unit assertions.
- **CLI tests:** patch the imported baseline/use-case symbols in `test_compress.py`; assert the color-VQ baseline is constructed with the codebook and appears in the comparison list, and that the full `color_sequence` pickle is not used as the artifact.
- **Boundary tests:** `pytest-archon` rules in `tests/colors_of_meaning/test_synesthetic_architecture.py` must stay green — domain imports no `sklearn`/`torch`/`ot`; the color-VQ codec lives in infrastructure (or in application using only `domain` + `shared`); `delta_e` is a `shared` leaf legal to import anywhere.
- **CDCT:** none required — no service crosses a process boundary in this step.
- **Verification:** `tox` for all 8 gates (never `pytest` alone); `tox -- tests/colors_of_meaning/...` for fast TDD loops; `tox -e format` before completion. 100% coverage including the PQ tiny-input fallback branch and the resolved `bits_per_dimension` path.

## Observability Plan
- Emit one structured log entry (with `correlation-id`) per compression run summarising method, counts, `compressed_size_bits`, `original_size_bits`, `compression_ratio`, `reconstruction_error` — the run's rate-distortion point. Do not log inside the per-color ΔE loop. Test asserts the summary log is emitted once per run.
- PQ: log once that the held-out split was used (train size, holdout size, seed) so out-of-sample distortion is reproducible; test asserts the split-summary log is emitted.
- No new metrics/tracing in the inner loops; existing use-case tracing (if any) is retained.

## Risks and Mitigations
- **Risk:** the three methods compress *different originals* (color-VQ: Lab colors; gzip/PQ: 384-dim embeddings), so a single "compression ratio" axis is apples-to-oranges. → **Mitigation:** SPEC Open Question forces an explicit decision — either normalise the original (e.g. all relative to the source embedding/text) or report each on a clearly-labelled axis; tests assert each method's own original-size formula rather than cross-method ratio equality.
- **Risk:** ΔE needs the *continuous* pre-quantization colors, which `ColoredDocument` does not store today. → **Mitigation:** decode-path Open Question in task 1/2 pins whether to thread continuous colors into the codec, add a pure `reconstructed_colors` decode helper, or define distortion index-to-centroid; the chosen input shape is fixed before tests are written.
- **Risk:** PQ train/holdout split changes existing reconstruction numbers and could fail size/ratio tests. → **Mitigation:** update `test_pq_compression_baseline.py` in task 7; keep size accounting consistent and assert held-out behaviour explicitly; deterministic seed makes tests stable.
- **Risk:** O(num_bins) Python `quantize` over a 4096-color codebook per color is slow for large corpora. → **Mitigation:** vectorisation is owned by `007-p1-1-learned-vq-codebook`; here keep inputs test-sized and note the dependency; do not duplicate the optimisation.
- **Risk:** branching in PQ split / use-case accounting trips `xenon`/`radon` complexity limits. → **Mitigation:** extract expressively-named private helpers (`_train_holdout_split`, `_mean_delta_e`, `_fixed_width_code_bits`) so public methods stay straight-line.
- **Risk:** removing tests for the deleted fabricated ratio/RLE drops coverage below 100% if replacements miss a branch. → **Mitigation:** task 7 maps each new honest path (zero-error, coarser-codebook, shrink, original-size, PQ fallback, resolved `bits_per_dimension`) to a test before deletion; `tox` coverage gate enforced.
- **Risk:** the pickle-artifact removal breaks `compress.py` consumers expecting the old file. → **Mitigation:** CLI is the only consumer; update `test_compress.py` and the print/serialise path together; no API/DTO surface depends on it.
