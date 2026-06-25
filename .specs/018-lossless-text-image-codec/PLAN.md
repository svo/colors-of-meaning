# Plan: Lossless text-to-image codec (decode an A4 image back into the exact source text)

## Implementation Strategy

Build a model-free, lossless color-barcode codec behind a new `DataImageCodec`
domain port, with a pure standard-library payload module (compression + page
framing) and a Pillow "P"-mode adapter that paints the bytes as palette cells on
an exact A4 canvas. Two thin use cases and two model-free CLIs wrap it. The codec
reuses the existing pure cell geometry (`cell_geometry.tile_grid`/`cell_center`)
and the exact-A4 / DPI discipline from feature 017, and adds no third-party
dependency (`pillow`/`numpy` already declared; `zlib`/`struct`/`binascii` are
stdlib).

The losslessness rests on two facts established with the library docs: PNG is a
lossless container, and a Pillow "P" image stores each pixel as a 1-byte palette
**index** so `numpy.asarray()` returns the indices verbatim — the index *is* the
byte. Encoding therefore never goes `Lab → RGB → Lab`, so the out-of-gamut
clipping that made feature 017's perceptual decode only approximate is absent;
the round-trip is exact by construction. A proof-of-concept already demonstrated
a whole 748k-character book encoded into one A4 "P"-mode PNG and decoded back
**byte-identical**.

Three design decisions keep it clean, fast, and testable:

1. **Isolate the pure parts.** Compression and page framing
   (`magic|version|page_index|page_count|len|crc32|payload`) live in a
   framework-free domain module that is unit-tested without Pillow or a file —
   exactly how 017 isolated `cell_geometry`. The Pillow open/save is the only
   impure surface and is exercised with a real `tmp_path`/`BytesIO` round-trip
   (017's lesson: never mock the imaging library — it hides round-trip bugs).
2. **Vectorize the raster.** Encode builds a `(columns, rows)` index grid with
   `Image.frombytes("P", …)` and upscales to A4 with `Resampling.NEAREST` — no
   per-cell `ImageDraw.rectangle` loop (which would be slow for 10^5 cells and
   push xenon past rank A). Decode reads indices with one `np.asarray` and a
   NEAREST downscale.
3. **Exact boundary, not a sentinel.** The per-page `payload_len` header bounds
   the data precisely, so trailing padding cells are dropped exactly — replacing
   017's white-sentinel trim and removing any palette-collision concern.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

- `domain/service/data_image_codec.py` → `DataImageCodec(ABC)` with
  `encode(text, output_path, dpi) -> List[str]` and `decode(input_paths) -> str`.
- `domain/service/data_payload.py` (stdlib only): `compress_text(text) -> bytes`,
  `decompress_text(blob) -> str`, `frame_page(payload, page_index, page_count)
  -> bytes`, `parse_page(framed) -> Page` (namedtuple `(page_index, page_count,
  payload)`; raises on bad magic / CRC32 mismatch / short buffer),
  `split_into_pages(blob, capacity) -> List[bytes]`, `reassemble(pages) -> bytes`
  (orders by `page_index`, validates `page_count` and uniqueness, concatenates).
- `domain/service/cell_geometry.py`: add pure `a4_canvas_pixels(dpi) ->
  Tuple[int, int]` (A4 mm→px). Leave feature 017's renderer untouched.
- No existing domain model is modified. Tests cover framing round-trips, every
  raise branch, page split/reassemble, and `a4_canvas_pixels`.

### Application Layer (`src/colors_of_meaning/application/`)

- `EncodeTextToImageUseCase(codec)` → `execute(text, output_path, dpi) ->
  List[str]`, logging `{chars, compressed_bytes, pages, dpi, cell_size,
  output_paths}`.
- `DecodeImageToTextUseCase(codec)` → `execute(input_paths) -> str`, logging
  `{pages_read, recovered_chars, crc_ok}`.
- Both depend only on the injected `DataImageCodec`.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

- `infrastructure/visualization/pillow_data_image_codec.py` →
  `PillowDataImageCodec(DataImageCodec)`. Constants: A4 mm / DPI default,
  default `cell_size` px, a deterministic 256-entry RGB **data palette**, and a
  16-byte page header layout. `encode`: compress → `split_into_pages` by the
  per-page cell capacity (`columns*rows - header_cells`) → `frame_page` each →
  build the `(cols, rows)` "P" grid via `frombytes` → `putpalette` → NEAREST
  upscale to `a4_canvas_pixels(dpi)` → `save(dpi=…)`; return the page paths
  (`output_path` for one page, `_pNN`-suffixed for many). `decode`: per path,
  `np.asarray(Image.open)` → NEAREST downscale to `(cols, rows)` → row-major
  bytes → `parse_page`; then `reassemble` → `decompress_text`. Extract
  `_paint_page`, `_read_page`, `_page_paths`, `_grid_shape` helpers to keep every
  method xenon rank A.

### Interface Layer (`src/colors_of_meaning/interface/`)

- `interface/cli/encode_lossless.py` and `interface/cli/decode_lossless.py` (tyro
  `@dataclass`, defaults, `main(args)`, `__main__` guard). Build
  `PillowDataImageCodec` + the use case at call time. **No projector/codebook.**
  `encode_lossless` resolves `--text` else reads `--input-path` whole;
  `decode_lossless` expands `--input-paths` (single / comma-list / glob), prints
  recovered length, writes `--output-path` or echoes stdout.
- `tox.ini`: add `[testenv:encode_lossless]` and `[testenv:decode_lossless]`.
- `README.MD`: add "Encode a Document Losslessly" contrasting it with the
  perceptual A4 image (lossy, semantic) vs this (lossless, model-free, color
  barcode).
- Architecture test: add the two CLIs to the CLI→use-case rule in
  `tests/colors_of_meaning/test_synesthetic_architecture.py`, and add an arch
  assertion that the domain `data_payload`/`data_image_codec` modules do **not**
  import `PIL`.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes.

## Dependency Injection

Use cases take the `DataImageCodec` port via `__init__`. CLIs construct
`PillowDataImageCodec` directly, as `encode_image.py` constructs its renderer. No
Lagom container registrations change; the API container is untouched.

## Task List

1. [ ] domain: `data_payload.py` (compress/decompress, frame/parse with CRC32,
   split/reassemble) + `a4_canvas_pixels` in `cell_geometry`. Tests: framing
   round-trip, CRC/magic/short-buffer raises, split boundary, reassemble ordering
   + missing/duplicate/count-mismatch raises.
2. [ ] domain: `DataImageCodec` port + ABC tests (no-instantiate, method
   presence, concrete subclass).
3. [ ] infrastructure: `PillowDataImageCodec.encode` (single page) — "P"-grid →
   NEAREST upscale → exact A4 + DPI save; deterministic.
4. [ ] infrastructure: `decode` — `np.asarray` indices → downscale → parse →
   decompress. Real `tmp_path` round-trip exactness (ASCII, Unicode, empty).
5. [ ] infrastructure: multi-page split + `_pNN` paths; any-order decode;
   missing/duplicate/CRC failures raise; page cap raise.
6. [ ] application: `EncodeTextToImageUseCase` + `DecodeImageToTextUseCase` with
   correlation-id logging + tests.
7. [ ] interface: `encode_lossless` / `decode_lossless` CLIs + tox envs + README +
   architecture-test wiring (CLI→use-case, domain-not-import-PIL).
8. [ ] tests: whole-book (`reports/austen_pride.txt`) byte-exact round-trip; A4
   dims + DPI; determinism; all error paths; 100% coverage.
9. [ ] run `tox`; confirm 8 gates + 100% coverage; run the real CLIs end-to-end
   (encode a book, decode it, diff against the original) and inspect a page PNG.

## Testing Strategy

House rules apply: one logical assertion per test, `test_should_..._when_...`
names, no network, prefer a real in-memory/`tmp_path` round-trip over mocking
Pillow. Key tests:

- **Payload (pure):** `frame_page`→`parse_page` returns the same payload; a
  flipped byte / wrong magic / truncated buffer raises; `split_into_pages` chunks
  to the exact capacity and the boundary case (compressed size == capacity) uses
  the minimal page count; `reassemble` orders by `page_index` and raises on a
  missing index, a duplicate, or a `page_count` mismatch. No Pillow needed.
- **Round-trip exactness:** encode→decode via real `tmp_path` PNGs equals the
  original for ASCII, Unicode/emoji, whitespace-heavy, and empty text — proving
  the real Pillow "P"-mode path, not a mock.
- **Whole book:** `reports/austen_pride.txt` (or a sizeable fixture) round-trips
  byte-identical across multiple pages decoded **in shuffled order**.
- **A4 fidelity:** each page's `size` equals the exact pixel dimensions for the
  DPI and `info["dpi"]` is present and correct.
- **Determinism:** two encodes of the same text/dpi/cell-size yield byte-identical
  PNG(s).
- **Failure modes:** decoding a page set with a missing/duplicate page, a
  corrupted cell, or a CRC mismatch raises `ValueError` (never silent wrong text).
- **Model-free:** encode/decode succeed with no projector or codebook on disk.
- **Use-case + CLI branches:** mock the codec; cover encode/decode delegation,
  `--text` vs `--input-path`, multi-path expansion, and stdout vs `--output-path`.

## Observability Plan

`correlation-id` structured logging in both use cases: encode logs `{chars,
compressed_bytes, pages, dpi, cell_size, output_paths}`; decode logs
`{pages_read, recovered_chars, crc_ok}`. No new metrics or tracing.

## Risks and Mitigations

- **PNG re-encoded to RGB by a third tool loses the "P" indices.** Mitigation:
  state the digital-PNG-preserved contract explicitly (as 017 does). Optionally a
  best-effort RGB fallback maps each cell to the nearest palette color, but exact
  recovery assumes the "P" PNG is preserved.
- **NEAREST resize index fidelity.** Mitigation: build the grid at `(cols, rows)`
  and use integer-aligned NEAREST upscale/downscale; prove fidelity with a real
  `tmp_path` round-trip test (the PoC already round-tripped a whole book exactly).
- **Performance at 10^5+ cells.** Mitigation: vectorized `frombytes`/`asarray`,
  no per-cell `ImageDraw` loop.
- **xenon complexity.** Mitigation: split encode/decode into `_paint_page` /
  `_read_page` / `_grid_shape` / `_page_paths` helpers; keep `data_payload`
  functions single-purpose — every block rank A.
- **Silent capacity truncation.** Mitigation: split into the minimal page count
  and `log()` it; above a configurable page cap, raise (no silent cap).
- **Port placement.** Mitigation: a dedicated `DataImageCodec` port keeps the
  color-as-data work off the semantic `DocumentImageRenderer`, with an arch test
  asserting the domain payload/codec modules never import `PIL`.
- **Scope creep into print/scan, encryption, API.** Mitigation: all deferred in
  Open Questions; v1 is digital, unencrypted, CLI-only.
