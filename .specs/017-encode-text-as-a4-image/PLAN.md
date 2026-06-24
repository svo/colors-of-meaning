# Plan: Encode a text as an A4 "colors of meaning" image (perceptual, with approximate decode)

## Implementation Strategy

Build one new Pillow-backed renderer behind a new domain port, two thin use
cases, and two CLIs, reusing the entire existing encode pipeline and the
bidirectional `lab_utils`. The encode direction needs no new color math
(`ColorCodebook.get_color` + `lab_to_rgb`), and neither does decode
(`rgb_to_lab` + `ColorCodebook.quantize`); `pillow` is already a declared
dependency, so nothing new is added to `install_requires`.

The image is a fixed-size Pillow canvas (exact A4 pixels for the chosen DPI),
drawn with `ImageDraw.rectangle` per cell and saved with `dpi=(d, d)` so the PNG
carries true A4 print size. The renderer never crops to content (unlike the
matplotlib `_save_figure`'s `bbox_inches="tight"`), so dimensions stay exact.

Three layouts share the canvas and the per-cell color machinery:

1. `score` (recoverable) — one cell per sentence in reading order, row-major on a
   fixed grid derived from canvas size and a cell size; this is the mode the
   decoder reads.
2. `signature` (aesthetic) — the histogram's top-N codebook colors as
   proportional horizontal bands filling the page.
3. `mosaic` (aesthetic) — the histogram as a grid of codebook colors tinted by
   frequency, scaled to the canvas.

Colors are **perceptual**. Decode of a `score`/`mosaic` image samples each cell's
representative color (cell-center or cell-mean), maps it through `rgb_to_lab` +
`codebook.quantize`, and reconstructs the histogram, which feeds the existing
`CompareDocumentsUseCase` for nearest-document retrieval. The round-trip is exact
for a coarse in-gamut codebook and approximate for the full 4096 codebook; the
acceptance contract is "recovers the source document by retrieval", not
"bit-exact".

A core design decision is to isolate the **pure** parts so they are testable
without a model or a real file: (a) cell geometry (canvas + cell size → list of
cell rectangles, row-major), and (b) the bin↔RGB mapping. The Pillow file I/O
(open/save) is the only impure surface and is exercised with a real in-memory
image (`BytesIO`), per the lesson that mocking the imaging library hides
round-trip bugs.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

Add `domain/service/document_image_renderer.py` → `DocumentImageRenderer(ABC)`
with `render_document_image(...)` and `decode_document_image(...) -> List[int]`,
and a domain-level `Literal` for the layout. Add a pure, framework-free helper
module for cell geometry (canvas size + cell size → ordered cell boxes) so the
layout math is unit-testable in the domain without Pillow; it depends only on
stdlib. No existing domain model is modified.

### Application Layer (`src/colors_of_meaning/application/`)

Add `EncodeDocumentToImageUseCase` (delegates to `EncodeDocumentUseCase` then the
renderer) and `DecodeImageToDocumentUseCase` (renderer decode →
`ColoredDocument.from_color_sequence` → optional `CompareDocumentsUseCase`
retrieval). Both depend only on injected ports/use cases.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

Add `infrastructure/visualization/pillow_document_image_renderer.py` →
`PillowDocumentImageRenderer(DocumentImageRenderer)`. A4 geometry constants
(mm dimensions, √2 ratio, per-DPI pixel size, default cell size) live here.
Render uses `Image.new` / `ImageDraw.Draw().rectangle` / `Image.save(dpi=...)`;
decode uses `Image.open` → `np.asarray` → per-cell sampling → `rgb_to_lab` +
`codebook.quantize`. Keep each method low-complexity (extract per-layout drawing
and cell-sampling helpers) to satisfy xenon rank A.

### Interface Layer (`src/colors_of_meaning/interface/`)

Add `interface/cli/encode_image.py` and `interface/cli/decode_image.py` (tyro
`@dataclass` args with defaults, `main(args)`, `__main__` guard, dependencies
built at call time with the codebook-`None` → `FileNotFoundError` guard). Add
`[testenv:encode_image]` and `[testenv:decode_image]` to `tox.ini`. Add a short
"Encode a Document as an A4 Image" section to `README.MD` and the tox-env table.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes. `lab_utils.lab_to_rgb` and `lab_utils.rgb_to_lab` are the complete
color contract.

## Dependency Injection

Use cases take their collaborators via `__init__` (renderer port, encode use
case, compare use case). CLIs construct `PillowDocumentImageRenderer`,
`PyTorchColorMapper` + `FileColorCodebookRepository` + `QuantizedColorMapper` +
`EncodeDocumentUseCase`, exactly as `encode.py`/`visualize.py` do. No Lagom
container registrations change; the API container in `interface/api/main.py` is
untouched.

## Task List

1. [ ] domain: add `DocumentImageRenderer` port + layout `Literal`, and the pure
   cell-geometry helper (canvas + cell size → ordered cell boxes). Tests for the
   geometry (exact box count, row-major order, exact A4 tiling).
2. [ ] infrastructure: implement `PillowDocumentImageRenderer.render_document_image`
   for `score` (perceptual cells via `get_color` + `lab_to_rgb`), with exact A4
   canvas and `dpi` metadata; keep helpers small (xenon A).
3. [ ] infrastructure: implement `decode_document_image` (open → `np.asarray` →
   per-cell sample → `rgb_to_lab` → `codebook.quantize` → `List[int]`).
4. [ ] infrastructure: add `signature` and `mosaic` layouts on the same canvas.
5. [ ] application: `EncodeDocumentToImageUseCase` and
   `DecodeImageToDocumentUseCase` (+ retrieval via `CompareDocumentsUseCase`).
6. [ ] interface: `encode_image` and `decode_image` CLIs + tox envs + README.
7. [ ] tests: round-trip exactness on `create_uniform_grid(2)`; A4 dimensions +
   DPI metadata; in-memory `BytesIO` real round-trip; retrieval returns the
   source document; deterministic `signature`/`mosaic`; codebook-absent error.
8. [ ] run `tox`; confirm 8 gates + 100% coverage; run the real CLIs end-to-end
   on a sample document and verify the printed A4 PNG visually.

## Testing Strategy

House rules apply: one logical assertion per test, `test_should_..._when_...`
names, mock the embedding model / heavy deps, no network. Key tests:

- **Geometry (pure):** cell-box count, row-major ordering, and that boxes tile
  the exact A4 canvas with no crop — no Pillow needed.
- **Round-trip exactness:** with `create_uniform_grid(2)` (8 well-separated,
  near-in-gamut colors) encode a known `color_sequence` to a real in-memory
  `score` image (`BytesIO`) and assert `decode_document_image` returns the exact
  sequence. This proves encode and decode against the real Pillow path, not a
  mock.
- **A4 fidelity:** assert the saved image's `size` equals the exact pixel
  dimensions for the DPI and that `info["dpi"]` is present and correct.
- **Retrieval acceptance:** a small seeded corpus of `ColoredDocument`s; decode a
  `score` image of one of them and assert the top-1 nearest neighbour is the
  source (perceptual approximate decode still retrieves).
- **Determinism:** `signature`/`mosaic` produce byte-identical output for
  identical input.
- **Fail-closed:** absent codebook → `FileNotFoundError`.
- **Use-case delegation + CLI branches:** mock the renderer/use cases; cover the
  layout dispatch, `--index` selection, and the decode/retrieve path to 100%.

Where Pillow is used, prefer a real `BytesIO` round-trip over mocking
`Image`/`ImageDraw`, so the decoder is genuinely exercised.

## Observability Plan

Add `correlation-id` structured logging in both use cases: encode logs
`{document_id, sentences, layout, dpi, canvas_px, output_path}`; decode logs
`{recovered_bins, top_match_id, top_distance}`. No new metrics or tracing.

## Risks and Mitigations

- **Approximate decode (out-of-gamut clipping).** Mitigation: state the
  perceptual/approximate contract explicitly; make the exactness test use a
  coarse in-gamut codebook where the round-trip is exact, and make the
  full-codebook acceptance retrieval-based (top-1 match) rather than
  sequence-exact. An `--exact` index mode is recorded as a future option.
- **A4 fidelity lost to cropping.** Mitigation: draw on a fixed exact canvas and
  save with `dpi`; never use `bbox_inches="tight"`; assert exact pixel size in a
  test.
- **Coverage of Pillow drawing/decoding.** Mitigation: round-trip a real small
  image in memory (`BytesIO`) to exercise draw + decode rather than asserting on
  mocks; this both covers the lines and proves runtime behaviour (the prior
  task's lesson that mocked-renderer 100% coverage misses real crashes).
- **xenon complexity.** Mitigation: extract per-layout draw helpers and the
  cell-sampling/geometry helpers so every function stays rank A (the corpus
  renderer needed the same split).
- **Port placement.** Mitigation: a dedicated `DocumentImageRenderer` port keeps
  the change off the analytical `FigureRenderer` ABC, avoiding the test-double /
  adapter ripple that extending an existing ABC causes.
- **Scope creep into API / real-scan / exact mode.** Mitigation: all three are
  explicitly deferred in Open Questions; v1 is CLI-only, perceptual, digital
  round-trip.
