# Feature: Encode a text as an A4 "colors of meaning" image (perceptual, with approximate decode)

## Overview

This feature turns a single text into a single **A4-format image** that is a
visual encoding of the document's semantic color distribution, and adds a
matching **decoder** that reads such an image back into a color histogram and
matches it to the source document. It is the physical-artifact expression of the
project thesis (documents as color distributions): a printable page that *is* a
document's "colors of meaning".

The encode path reuses the existing pipeline end to end. A document is already
turned into an **ordered** sequence of codebook bins:
`text → SentenceEmbeddingAdapter.encode_document_sentences` (per sentence) →
`QuantizedColorMapper.embed_batch_to_bins`
(`src/colors_of_meaning/domain/service/color_mapper.py:48`) → a `color_sequence:
List[int]` that is already carried on `ColoredDocument`
(`src/colors_of_meaning/domain/model/colored_document.py:10`, built by
`from_color_sequence`, `:59`). `EncodeDocumentUseCase.execute`
(`application/use_case/encode_document_use_case.py:12`) already returns a
`ColoredDocument` with both `histogram` and `color_sequence` populated. Rendering
a bin to a pixel reuses `ColorCodebook.get_color`
(`domain/model/color_codebook.py:30`) plus `lab_to_rgb`
(`shared/lab_utils.py:39`). The **decode** direction reuses the inverse that
already exists: `rgb_to_lab` (`shared/lab_utils.py:7`) then
`ColorCodebook.quantize` (`color_codebook.py:25`). No new color math is required,
and `pillow` is already a declared dependency (`setup.cfg` `install_requires`),
so no new third-party dependency is introduced.

The image is rendered with **perceptual** colors: every cell is the true
`Lab → RGB` color of its codebook bin, so the page is an honest "colors of
meaning" artifact. Because the codebook is a uniform Lab grid that includes
points outside the sRGB gamut, `lab_to_rgb` clips, so the perceptual round-trip
is **approximately** decodable, not bit-exact: re-quantizing a cell can land on a
neighbouring bin. This is sufficient for the headline use — re-reading a page
recovers a histogram close enough to retrieve the source document — and the
original *text* is intentionally not recoverable (the 384-dim → 3-dim projection
is lossy by design). A future `--exact` index-encoding mode (lossless) is out of
scope (see Open Questions).

Two encoding **modes** are provided behind a single layout flag, sharing one
Pillow canvas:

- `score` (recoverable): one cell per sentence, in reading order, row-major on a
  fixed grid. This is the literal "encode the text"; it is the mode the decoder
  reads.
- `signature` (aesthetic): the document histogram's dominant codebook colors as
  proportional bands filling the page — the single-document, A4, perceptual
  cousin of the corpus-signature renderer.
- `mosaic` (aesthetic/deterministic): the document histogram laid out as a grid
  (each codebook color tinted by its frequency) scaled to fill A4.

A4 fidelity is exact: the canvas is drawn at an exact pixel size (210×297 mm,
ratio √2; 2480×3508 px at 300 DPI) and saved with DPI metadata so it prints to
true A4. The renderer must NOT crop to content — the existing matplotlib
`_save_figure` uses `bbox_inches="tight"`
(`infrastructure/visualization/matplotlib_figure_renderer.py`), which would
destroy the A4 dimensions; the new Pillow adapter draws on a fixed canvas and
avoids the problem entirely.

This feature adds a **new domain port** `DocumentImageRenderer` rather than
extending `FigureRenderer`. `FigureRenderer` is the matplotlib analytical-figure
port (palette / histograms / t-SNE / confusion / corpus-signature); an A4
document artifact is a distinct, Pillow-based, round-trippable responsibility,
and extending that ABC would force its test double and `MatplotlibFigureRenderer`
to implement an unrelated method.

## User Stories

- As a researcher, I want to turn a document into one printable A4 "colors of
  meaning" image so I have a physical, shareable artifact of its semantic color
  distribution.
- As a researcher, I want the ordered `score` image to be decodable back into a
  color histogram and matched to its source document, demonstrating that the
  perceptual encoding preserves *retrievable* meaning even though the text itself
  is not recoverable.
- As a communicator, I want aesthetic `signature` and `mosaic` A4 posters of a
  document's color distribution for presentation.
- As a maintainer, I want the image to print at true A4 size (exact pixels plus
  DPI metadata), not a content-cropped approximation.
- As a contributor, I want the mechanism to reuse the existing encode pipeline
  and the bidirectional `lab_utils`, and to live behind a clean new port rather
  than bolted onto the analytical `FigureRenderer`.

## Acceptance Criteria

- [ ] Given a text and `--layout score`, when `encode_image` runs, then it writes
  a PNG with exact A4 pixel dimensions for the chosen DPI (2480×3508 at 300 DPI)
  and embedded DPI metadata, with cells tiling the exact canvas (never cropped).
- [ ] Given a `score` image and a corpus of encoded documents, when
  `decode_image` runs, then it recovers a color histogram and returns the source
  document as the top-1 nearest neighbour by color distance.
- [ ] Given a coarse, in-gamut codebook (e.g. `create_uniform_grid(2)`), when a
  known `color_sequence` is encoded to a `score` image and decoded, then the
  recovered sequence equals the original exactly (perceptual decode is exact when
  cell colors are in-gamut and well separated).
- [ ] Given `--layout signature` or `--layout mosaic`, when `encode_image` runs
  twice on the same document, then it writes byte-identical deterministic A4
  images of the document's color distribution.
- [ ] Given the codebook artifact is absent, when `encode_image` runs, then it
  fails closed with `FileNotFoundError` ("Codebook not found"), consistent with
  `encode.py` / `visualize.py`.
- [ ] Given the decode samples each cell, when a cell color does not round-trip
  exactly (out-of-gamut clipping), then it maps to the nearest codebook bin
  (approximate, documented contract), and the recovered histogram stays within a
  stated tolerance on the test fixtures.
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays
  100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New port `domain/service/document_image_renderer.py` →
`DocumentImageRenderer(ABC)`: `render_document_image(document, codebook, layout,
output_path, dpi)` and `decode_document_image(input_path, codebook) ->
List[int]`. The port declares the abstraction; file I/O happens in the adapter,
mirroring how `FigureRenderer`'s render methods already perform I/O in their
implementation. The layout choices are a domain-level enum/`Literal`
(`score` / `signature` / `mosaic`). No new color math: encode uses
`ColorCodebook.get_color` + `lab_to_rgb`; decode uses `rgb_to_lab` +
`ColorCodebook.quantize`. `ColoredDocument` and `ColorCodebook` are read, not
modified.

### Application Layer (`src/colors_of_meaning/application/`)

New `EncodeDocumentToImageUseCase`: orchestrates `text → ColoredDocument` (via the
existing `EncodeDocumentUseCase`) → `DocumentImageRenderer.render_document_image`.
New `DecodeImageToDocumentUseCase`: `image → DocumentImageRenderer.decode_* →
ColoredDocument.from_color_sequence →` (optional) nearest-document retrieval via
the existing `CompareDocumentsUseCase`. Both depend only on ports.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/visualization/pillow_document_image_renderer.py` →
`PillowDocumentImageRenderer(DocumentImageRenderer)`. Render: `Image.new("RGB",
(W, H))`, `ImageDraw.rectangle` per cell, `Image.save(path, dpi=(d, d))`. Decode:
`np.asarray(Image.open(path))`, sample each cell's representative color, map via
`rgb_to_lab` + `codebook.quantize`. A4 geometry constants (mm, ratio, per-DPI
pixel size) live here as rendering detail. Pillow is already installed
(`pillow==12.2.0`).

### Interface Layer (`src/colors_of_meaning/interface/`)

New CLIs `interface/cli/encode_image.py` and `interface/cli/decode_image.py`,
each a tyro `@dataclass` with all-default fields and a `main(args)`/`__main__`
guard, instantiating dependencies at call time (the `visualize.py` / `encode.py`
pattern). New `[testenv:encode_image]` and `[testenv:decode_image]` tox envs.
README gains a short usage section. No API endpoint (see Open Questions).

### Shared Layer

No code changes. `shared/lab_utils.py` already provides both `lab_to_rgb`
(`:39`) and `rgb_to_lab` (`:7`), which are the entire encode/decode color
contract.

## API Contracts

No API contract changes in this feature. A binary image response is not a
Pydantic DTO, which conflicts with the interface rule that every endpoint
response is a DTO; an HTTP `POST /encode/image` is deferred (Open Questions). The
existing `POST /query/palette` contract is unaffected.

## CLI Impact

Two new CLIs, both requiring pre-built projector + codebook artifacts:

- `encode_image`: `--dataset-path` (text file, one document per line; encodes the
  first/`--index` document) | `--text`, `--layout {score,signature,mosaic}`
  (default `score`), `--dpi` (default 300), `--config`, `--model-path`,
  `--codebook-name`, `--output-path` (default `reports/figures/document_a4.png`).
- `decode_image`: `--image-path`, `--codebook-name`, `--encoded-documents`
  (corpus for retrieval), `--k` (default 5). Prints the recovered histogram
  summary and the top-k nearest documents.

No existing CLI behaviour changes.

## Dependency Injection

The two use cases receive a `DocumentImageRenderer` (and, for encode, the
`EncodeDocumentUseCase` / `QuantizedColorMapper` stack; for decode, the
`CompareDocumentsUseCase`) via their constructors. The CLIs build the concrete
`PillowDocumentImageRenderer` and the encode stack at call time, matching
`visualize.py`/`encode.py`. No Lagom container (the API container) is modified,
since the CLIs wire directly and the API is untouched.

## Observability

Structured logging with `correlation-id` in both use cases: encode logs document
length (sentences), layout, dpi, output path, and canvas size; decode logs the
recovered bin count, histogram entropy, and the top match id. Consistent with the
logging style in `interface/api/main.py`. No new metrics/tracing required.

## Open Questions

- Should an HTTP `POST /encode/image` be added? It returns binary, which is not a
  Pydantic DTO; options are a binary `Response` (bends the DTO rule), a
  base64/path DTO, or leaving it CLI-only. Default: CLI-only for this feature.
- Should an `--exact` index-encoding mode (lossless, scan-robust, arbitrary
  palette) be added alongside the perceptual default? Default: out of scope now;
  perceptual only.
- Should the image survive a real camera scan (fiducial corner markers, de-skew;
  e.g. a Data Matrix anchor via `pylibdmtx`)? Default: digital round-trip only;
  real-scan robustness is future work.
- Cell geometry defaults: cell size, number of columns, and whether to render a
  self-describing header row encoding `(sentence_count, codebook_id)` so a decoder
  needs no out-of-band geometry. Default: fixed, documented geometry; no header
  row in v1.
- Should `decode_image` require a corpus (`--encoded-documents`) for retrieval,
  or also support emitting just the recovered histogram when no corpus is given?
  Default: retrieval when a corpus is supplied, histogram summary otherwise.
