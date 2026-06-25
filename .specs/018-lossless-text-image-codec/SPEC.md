# Feature: Lossless text-to-image codec (decode an A4 image back into the exact source text)

## Overview

This feature adds a **lossless** companion to the perceptual "colors of
meaning" image feature (`.specs/017-encode-text-as-a4-image/`). Where the
perceptual `score`/`signature`/`mosaic` layouts are *lossy by design* — the
`text → 384-dim embedding → projector → 3-dim Lab → 4096-bin quantization`
pipeline destroys the information that distinguishes one sentence from another,
so the **source text is intentionally not recoverable** — this feature encodes a
document so that, given the image (and its embedded palette), the **exact
original text is recovered byte-for-byte**.

The crucial distinction is *what the colors mean*. In feature 017 a cell's color
is a **semantic** signal (a lossy projection of meaning). Here a cell's color is
a **data symbol**: the image is a color barcode (the Microsoft HCCB / ISO-IEC
23634 JAB-Code family) and the palette is its alphabet. The colors carry the
document's *bytes*, not its *meaning*. No semantic projection is involved, so the
encoder is **model-free**: it needs neither a trained projector nor the Lab
`ColorCodebook`.

The encode path is: `text → UTF-8 → zlib (DEFLATE) → framed pages
[magic, version, page_index, page_count, length, CRC32, payload] → palette
symbols → cells on an exact A4 canvas → lossless PNG`. The decode path is the
exact inverse: `PNG → per-cell palette indices → framed pages → CRC-checked
reassembly → DEFLATE-inflate → UTF-8 → exact text`.

Losslessness is guaranteed for the **digital round-trip** (PNG in, PNG out — the
same operating assumption as feature 017's decoder) by two facts: PNG is a
lossless container, and a Pillow **"P" (palette) mode** image stores each pixel
as a 1-byte index into an embedded ≤256-color palette (the PNG `PLTE` chunk), so
`numpy.asarray()` of a "P" image returns the **indices** directly — the index
*is* the byte. There is no `Lab → RGB → Lab` round-trip and therefore none of the
out-of-gamut clipping that made feature 017's perceptual decode only
*approximate*. A trailing-padding boundary is set by an exact **length header**,
not the white sentinel that 017 uses.

This realizes the `--exact` index-encoding mode that feature 017 explicitly
deferred to its Open Questions ("lossless, scan-robust, arbitrary palette"). It
reuses the existing pure cell geometry (`cell_geometry.tile_grid`
`src/colors_of_meaning/domain/service/cell_geometry.py:6`, `cell_center` `:34`)
and the exact-A4 / DPI rendering discipline, and introduces no new third-party
dependency: `pillow` and `numpy` are already declared (`setup.cfg`
`install_requires`); `zlib`, `struct`, and `binascii` are standard library.

A document larger than one A4 page is split across the **minimal number of
pages**; each page self-describes its `(page_index, page_count)`, so pages can be
decoded in any order and a missing or duplicated page is detected rather than
silently producing wrong text. **Print-then-scan** robustness (fiducial markers,
color calibration swatches, Reed-Solomon error correction) is out of scope; v1 is
a digital round-trip, consistent with feature 017.

## Core Domain Concepts

- **Data palette**: a fixed, deterministic ordered list of 256 display colors;
  symbol `i` (a byte value `0–255`) renders as palette entry `i`. The palette is
  the decode alphabet/key and is embedded in every PNG (`PLTE`), so the artifact
  is self-describing. Distinct from the semantic Lab `ColorCodebook`: palette
  colors carry no meaning and need not be injective (decode reads indices, not
  colors).
- **Module / cell**: a solid square block of one palette color carrying one byte,
  laid out row-major on the A4 canvas via the existing cell geometry. Larger
  cells trade capacity for visual structure / future scan-robustness.
- **Page frame**: `MAGIC(4) | version(1) | page_index(2) | page_count(2) |
  payload_len(4) | crc32(4) | payload`, painted as the leading cells of a page.
  The length header bounds the payload exactly (trailing cells are padding).
- **Lossless / digital round-trip**: PNG-in/PNG-out recovers the exact bytes;
  byte-for-byte text equality is the contract, not "retrieval".

## User Stories

- As a researcher, I want to encode a document into a printable A4 color image
  and later recover the **exact original text** from that image, so the page is a
  self-contained, human-pretty backup of the document.
- As a researcher, I want the lossless encoder to be **model-free** (no trained
  projector, no Lab codebook), so I can archive arbitrary text without the
  semantic pipeline.
- As an archivist, I want a long document to span the **fewest** A4 pages and to
  decode correctly even if I feed the pages **out of order** or am **missing**
  one (it should tell me, not hand back corrupted text).
- As a maintainer, I want each page to print at true A4 size (exact pixels + DPI
  metadata) and be **deterministic** (same input → byte-identical PNG).
- As a contributor, I want this to reuse the existing cell geometry and live
  behind a clean new `DataImageCodec` port rather than being bolted onto the
  semantic `DocumentImageRenderer`.

## Acceptance Criteria

- [ ] Given any text and `encode`, when `decode` runs on the resulting page(s),
  then it returns the **byte-for-byte original text** — verified for ASCII, for
  Unicode/emoji, for whitespace-heavy text, and for an empty string.
- [ ] Given a text that fits one page, when `encode` runs, then it writes a single
  PNG with exact A4 pixel dimensions for the DPI (2480×3508 at 300 DPI) and
  embedded DPI metadata, never cropped.
- [ ] Given a text larger than one page, when `encode` runs, then it writes the
  **minimal number of pages**, returns their paths, and `decode` reconstructs the
  exact text when given those pages **in any order**.
- [ ] Given a whole public-domain book (e.g. `reports/austen_pride.txt`), when it
  is encoded and decoded, then the recovered text equals the original exactly.
- [ ] Given a page set with a missing page, a duplicated `page_index`, a
  mismatched `page_count`, a bad magic, or a CRC32 mismatch, when `decode` runs,
  then it raises a clear error (`ValueError`) rather than returning wrong text.
- [ ] Given the same text, dpi, and cell size, when `encode` runs twice, then it
  writes **byte-identical** PNG page(s).
- [ ] Given no trained projector and no Lab codebook on disk, when `encode`/`decode`
  run, then they succeed (the lossless codec is model-free and codebook-free).
- [ ] Given `tox` is run, then all eight quality gates pass and coverage stays
  100%.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

New port `domain/service/data_image_codec.py` → `DataImageCodec(ABC)`:
`encode(text: str, output_path: str, dpi: int) -> List[str]` (returns the written
page paths) and `decode(input_paths: List[str]) -> str`. The port declares the
abstraction; file I/O happens in the adapter, mirroring how feature 017's
`DocumentImageRenderer` performs I/O in its implementation. A dedicated port (not
an extension of `DocumentImageRenderer`
`src/colors_of_meaning/domain/service/document_image_renderer.py:10`) keeps the
color-as-data responsibility off the color-as-meaning ABC, exactly as 017 kept
its work off `FigureRenderer`.

New pure helper `domain/service/data_payload.py` (standard library only —
`struct`, `zlib`, `binascii`): `compress_text`/`decompress_text`,
`frame_page`/`parse_page` (magic + version + `page_index` + `page_count` +
`payload_len` + CRC32), `split_into_pages(blob, capacity)`, and
`reassemble(pages)`. Pure and framework-free so the framing/compression logic is
unit-testable without Pillow or any file, mirroring `cell_geometry`. Reuses
`cell_geometry.tile_grid`/`cell_center`; adds a pure `a4_canvas_pixels(dpi)` to
`cell_geometry` (A4 mm→px), leaving 017's renderer untouched. No domain model is
modified.

### Application Layer (`src/colors_of_meaning/application/`)

New `EncodeTextToImageUseCase` (delegates to `DataImageCodec.encode`) and
`DecodeImageToTextUseCase` (delegates to `DataImageCodec.decode`). Both are thin,
depend only on the injected port, and add `correlation-id` structured logging. No
embedding adapter, projector, codebook, or compare use case is involved.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

New `infrastructure/visualization/pillow_data_image_codec.py` →
`PillowDataImageCodec(DataImageCodec)`. Owns the 256-color **data palette**
constant and the A4 geometry. Encode: lay each page's framed bytes onto a
`(columns, rows)` grid (`Image.frombytes("P", (cols, rows), page_bytes)`),
`putpalette(...)`, upscale to the exact A4 canvas with `Image.Resampling.NEAREST`
(no per-cell `ImageDraw` loop — vectorized, low-complexity, blocky cells), and
`save(path, dpi=(d, d))`. Decode: `np.asarray(Image.open(path))` yields the
palette **indices** at full resolution; NEAREST-downscale to `(columns, rows)`
(or sample `cell_center` indices), flatten row-major to bytes, `parse_page`, and
reassemble. Pillow and numpy are already installed.

### Interface Layer (`src/colors_of_meaning/interface/`)

New CLIs `interface/cli/encode_lossless.py` and
`interface/cli/decode_lossless.py`, each a tyro `@dataclass` with all-default
fields and a `main(args)`/`__main__` guard, building the
`PillowDataImageCodec` + use case at call time. **No projector/codebook loading**
(model-free) — a notable simplification over `encode_image.py`. New
`[testenv:encode_lossless]` and `[testenv:decode_lossless]` tox envs. README gains
a short "Encode a Document Losslessly" section contrasting it with the perceptual
A4 image. No API endpoint (binary response is not a Pydantic DTO; see 017).

### Shared Layer

No code changes. (A4 pixel geometry is added to the existing pure
`domain/service/cell_geometry.py`, not to `shared`.)

## API Contracts

No API contract changes. As in feature 017, a binary image (or recovered text
blob) is not a Pydantic DTO; an HTTP endpoint is deferred (Open Questions). The
existing `POST /query/palette` contract is unaffected.

## CLI Impact

Two new CLIs, neither requiring any pre-built artifact:

- `encode_lossless`: `--text` | `--input-path` (a UTF-8 text file read whole as
  one document), `--output-path` (default `reports/figures/document_exact.png`;
  multi-page runs insert `_p01`, `_p02`, … before the extension), `--dpi`
  (default 300), `--cell-size` (px per module; default documented; density vs
  robustness). Prints the page count and written paths.
- `decode_lossless`: `--input-paths` (one path, a comma-separated list, or a glob
  of the page PNGs). Prints the recovered character count and writes/echoes the
  exact recovered text (`--output-path` optional; stdout otherwise).

No existing CLI behaviour changes.

## Dependency Injection

The two use cases receive a `DataImageCodec` via their constructors. The CLIs
build the concrete `PillowDataImageCodec` and the use case at call time, matching
the `encode_image.py`/`visualize.py` pattern. No Lagom container is modified; the
API is untouched.

## Observability

Structured logging with `correlation-id` in both use cases: encode logs
`{chars, compressed_bytes, pages, dpi, cell_size, output_paths}`; decode logs
`{pages_read, recovered_chars, crc_ok}`. Consistent with the logging style in
feature 017's use cases and `interface/api/main.py`. No new metrics/tracing.

## Open Questions

- **Palette as external sidecar vs embedded?** Default: embedded in the PNG
  `PLTE` (self-describing, no external key needed). An optional external palette
  artifact (so the colors are unreadable without the key) is a future option.
- **Print-then-scan robustness?** Default: digital round-trip only. Surviving a
  real camera scan needs fiducial corners + de-skew, on-page color-calibration
  swatches, and Reed-Solomon ECC over the byte stream (a color-2D-barcode build).
  Deferred, as in feature 017.
- **Cell/module size default and exposure.** Default: a fixed, documented module
  size with a `--cell-size` override; smaller = denser (fewer pages) but less
  robust. v1 targets digital fidelity.
- **Optional pre-compression encryption** (encrypt the DEFLATE blob so the
  artifact is a private backup). Out of scope now.
- **Integrate as `encode_image --layout exact` vs standalone CLIs?** Default:
  standalone, model-free CLIs — folding it into `encode_image` would drag in the
  projector/codebook wiring this feature deliberately avoids.
- **Maximum page cap.** Default: cap the page count and raise above it (no silent
  truncation); the cap is configurable.
