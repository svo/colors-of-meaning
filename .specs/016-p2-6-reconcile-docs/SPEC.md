# Feature: Reconcile docs (384-dim / ~1024:1; stop labeling the uniform grid "VQ")

## Overview

The project documentation overstates two claims that the implemented code does
not support, creating documentation drift between the prose and the source of
truth.

First, the **compression claim** is vague and aspirational. The implemented
embedding dimension is 384 (`src/colors_of_meaning/shared/synesthetic_config.py:9`,
`ProjectorConfig.embedding_dim = 384`), and a document histogram quantizes to a
4,096-color palette (`CodebookConfig.num_bins = 4096`,
`synesthetic_config.py:19`; `16 ** 3 = 4096`,
`src/colors_of_meaning/domain/model/color_codebook.py:44-45`). A 384-dim float32
embedding is `384 * 32 = 12,288` bits; a single 4,096-color code is
`log2(4096) = 12` bits. The exact ratio is therefore `12288 / 12 = 1024:1`. The
docs instead say generic, imprecise figures: README.MD:13 and
`.claude/CLAUDE.md:5` both say "extreme compression (1000x+)", and README.MD:20
says "12,288 bits/sentence to 12 bits" without ever stating the precise `~1024:1`
ratio. The blog post the docs reference
(https://www.qual.is/posts/colors-of-meaning) is reported to cite 768-dim /
2000:1; those figures do **not** appear anywhere in README.MD, `.claude/CLAUDE.md`,
or ROADMAP.md (verified: `grep -rn -i "768\|2000" README.MD .claude/CLAUDE.md
ROADMAP.md` returns no hits), so the in-repo dimension is already correct at 384
and only the compression ratio needs to be made precise and honest.

Second, the **"vector quantization" / "VQ" labeling** is misleading. The only
codebook constructor is `ColorCodebook.create_uniform_grid`
(`color_codebook.py:43-57`), which builds a **fixed, uniform** grid of Lab points
with `np.linspace` over L (0-100), a (-128-127), and b (-128-127). Quantization
(`color_codebook.py:19-29`) is nearest-neighbour assignment to this fixed grid.
This is a static, hand-specified palette, not a learned vector-quantization
codebook. Learned VQ does not exist until `007-p1-1-learned-vq-codebook` lands.
Until then, the docs should describe the mechanism as a "fixed/uniform Lab color
grid" with "nearest-color quantization", not "VQ" or "vector quantization". The
labels to soften are README.MD:21 ("color VQ"), README.MD:97 ("color VQ
compression"), README.MD:183 ("Color VQ compression analysis"), and
`.claude/CLAUDE.md:10` ("4,096-color palette used for vector quantization").
After `007-p1-1-learned-vq-codebook` lands, the "VQ" terminology becomes honest
and may be reinstated.

This is a documentation-only reconciliation step. No production code, CLI, API, or
test behaviour changes are required by the Definition of Done. The CLI flag value
`--method vq` (`src/colors_of_meaning/interface/cli/compress.py:27`,
`method: str = "vq"`) is a public interface string and is explicitly **out of
scope** for this docs step; renaming it would be a code change deferred to or after
`007-p1-1-learned-vq-codebook` (see Open Questions).

`.claude/CLAUDE.md` is the project instruction file. Per house rules it is treated
conservatively: this spec recommends the two specific, factual edits above
(precise `~1024:1` ratio at line 5; soften "vector quantization" at line 10) and
does **not** rewrite any mandatory rule, naming convention, or architectural
constraint in that file.

## User Stories

- As a researcher reading the README, I want the compression claim stated as the
  exact, reproducible `~1024:1` (12,288 bits to 12 bits) so that I can trust the
  headline number matches the 384-dim configuration the code actually ships.
- As a new contributor, I want the docs to describe the codebook as a fixed
  uniform Lab grid rather than "VQ" so that I am not misled into thinking a learned
  vector-quantization codebook already exists before
  `007-p1-1-learned-vq-codebook`.
- As a maintainer, I want the in-repo docs (README.MD, `.claude/CLAUDE.md`) to
  agree with `synesthetic_config.py` on dimension and compression so that the
  Definition of Done for P2-6 is objectively verifiable.
- As a reviewer, I want any divergence from the blog's 768-dim / 2000:1 figures
  explicitly acknowledged as "aspirational vs implemented" so that the discrepancy
  is intentional and documented rather than silent drift.

## Acceptance Criteria

- [ ] Given README.MD:13 states "extreme compression (1000x+)", when this step is
  complete, then it states the implemented compression precisely as `~1024:1`
  (derived from 384-dim float32 = 12,288 bits to a 12-bit, 4,096-color code).
- [ ] Given README.MD:20 lists "12,288 bits/sentence to 12 bits (4,096-color
  palette)", when this step is complete, then the line also states the explicit
  `~1024:1` ratio so the figure and the ratio are consistent.
- [ ] Given `.claude/CLAUDE.md:5` states "extreme compression (1000x+)", when this
  step is complete, then it states `~1024:1` consistently with README.MD, and no
  mandatory rule, naming convention, or architectural constraint elsewhere in
  `.claude/CLAUDE.md` is altered.
- [ ] Given README.MD:21, README.MD:97, and README.MD:183 label the mechanism
  "color VQ" / "Color VQ", when this step is complete, then they describe a
  "fixed uniform Lab color grid" with "nearest-color quantization" and cross-
  reference `007-p1-1-learned-vq-codebook` as the step after which learned "VQ"
  terminology becomes accurate.
- [ ] Given `.claude/CLAUDE.md:10` says the palette is "used for vector
  quantization", when this step is complete, then it describes nearest-color
  quantization against a fixed uniform Lab grid (not learned VQ), consistent with
  `ColorCodebook.create_uniform_grid`.
- [ ] Given the blog cites 768-dim / 2000:1, when this step is complete, then the
  README explicitly notes those as aspirational blog figures distinct from the
  implemented 384-dim / `~1024:1` (resolution of whether to keep this note is an
  Open Question; default is to add a one-line note).
- [ ] Given the docs reconciliation is complete, when `grep -rn -i "768\|2000:1\|
  2000x" README.MD .claude/CLAUDE.md` is run, then it returns no claim presenting
  768-dim or 2000:1 as the implemented figure (only an explicit "aspirational
  blog" note, if kept, may mention them).
- [ ] Given the docs reconciliation is complete, when `tox` is run, then all eight
  quality gates pass and coverage remains 100% (docs edits do not regress the
  build).

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. `domain/model/color_codebook.py` is read only as the source of truth
that `create_uniform_grid` (lines 43-57) is a fixed uniform grid, not learned VQ;
its code is not modified.

### Application Layer (`src/colors_of_meaning/application/`)

No changes.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No changes.

### Interface Layer (`src/colors_of_meaning/interface/`)

No code changes. The CLI flag value `--method vq`
(`interface/cli/compress.py:27`) is intentionally left unchanged; only README
prose describing the mechanism is softened. Renaming the flag is deferred (see
Open Questions) to avoid a breaking interface change during a docs step.

### Shared Layer

No code changes. `shared/synesthetic_config.py` is the source of truth
(`embedding_dim = 384`, line 9; `num_bins = 4096`, line 19) against which the docs
are reconciled; it is read, not modified.

## API Contracts

No API contracts change. No controller, route, or Pydantic DTO is added or
modified. The HTTP query-by-palette contract is unaffected.

## CLI Impact

No CLI behaviour changes. The `--method vq` default
(`interface/cli/compress.py:27`) and all other flags remain exactly as documented.
README prose around the compress CLI (README.MD:97, 183, 193-194) is reworded to
stop calling the fixed-grid mechanism "VQ"; the commands themselves are unchanged.

## Dependency Injection

No changes. No Lagom container registrations are added or modified.

## Observability

No changes. No new logging, metrics, or tracing. Documentation edits have no
runtime footprint.

## Open Questions

- Should README.MD retain an explicit "aspirational (blog) vs implemented" note
  recording the blog's 768-dim / 2000:1 alongside the implemented 384-dim /
  `~1024:1`? Default assumption: yes, a single clarifying line, so the discrepancy
  with https://www.qual.is/posts/colors-of-meaning is intentional and traceable
  rather than silent.
- Should a cheap config-to-docs consistency test be added (e.g. assert
  `ProjectorConfig().embedding_dim == 384` and that README.MD contains "1024" and
  not a "768"/"2000:1" implemented-figure claim)? A config-self-consistency test
  already exists (`tests/colors_of_meaning/shared/test_synesthetic_config.py:173`
  asserts `embedding_dim == 384`), but nothing pins the docs. Default assumption:
  add a small README-grep guard test only if it stays simple and keeps `tox`
  green; otherwise defer.
- The "VQ" CLI flag value `--method vq` (`interface/cli/compress.py:27`) is
  honest only after `007-p1-1-learned-vq-codebook`. Should it be renamed (a
  breaking interface change with test updates) or left as-is with softened docs?
  Default assumption for this docs step: leave the flag, soften only prose, and
  revisit the rename when 007 lands.
- Residual drift outside the stated scope: `embedding_dim = 768` still appears in
  `tests/colors_of_meaning/interface/cli/test_eval.py:144` and `:218`. P2-6 scope
  is README.MD + `.claude/CLAUDE.md`; should these stray 768 test values be
  corrected to 384 here for consistency, or deferred? Default assumption: out of
  scope for this docs step; flag for a follow-up unless the reviewer wants it
  folded in.
