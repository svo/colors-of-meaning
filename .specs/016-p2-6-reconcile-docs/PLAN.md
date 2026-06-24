# Plan: Reconcile docs (384-dim / ~1024:1; stop labeling the uniform grid "VQ")

## Implementation Strategy

This is a documentation-only reconciliation step, not a code feature. The work
edits two existing files in place (README.MD and `.claude/CLAUDE.md`) to make two
claims honest against the source of truth, then verifies `tox` stays green. No new
documentation files are created (per house rules), and no production code, CLI,
API, or DI wiring is touched.

Two corrections, each grounded in a verified location:

1. **Precise compression ratio.** Replace the vague "extreme compression
   (1000x+)" with the exact implemented figure `~1024:1`. The arithmetic is fixed
   by the source of truth: `ProjectorConfig.embedding_dim = 384`
   (`src/colors_of_meaning/shared/synesthetic_config.py:9`) and
   `CodebookConfig.num_bins = 4096` (`synesthetic_config.py:19`;
   `16 ** 3 = 4096`, `domain/model/color_codebook.py:44-45`). A 384-dim float32
   embedding is `384 * 32 = 12,288` bits; one 4,096-color code is
   `log2(4096) = 12` bits; `12288 / 12 = 1024`. Edit sites: README.MD:13,
   README.MD:20, and `.claude/CLAUDE.md:5`.

2. **Stop calling the fixed grid "VQ".** The only codebook constructor is
   `ColorCodebook.create_uniform_grid` (`color_codebook.py:43-57`), a fixed
   uniform `np.linspace` Lab grid with nearest-color quantization
   (`color_codebook.py:19-29`) — not learned vector quantization. Reword the "VQ"
   labels to "fixed uniform Lab color grid" / "nearest-color quantization" and
   cross-reference `007-p1-1-learned-vq-codebook`, after which learned "VQ"
   terminology becomes accurate. Edit sites: README.MD:21, README.MD:97,
   README.MD:183 (and the surrounding compress prose at README.MD:193-194), and
   `.claude/CLAUDE.md:10`.

Verified scoping facts: the blog's "768"/"2000" figures do **not** appear in
README.MD, `.claude/CLAUDE.md`, or ROADMAP.md (`grep -rn -i "768\|2000"` returns
no hits there), so the in-repo dimension is already 384 and only the ratio and the
"VQ" naming need correcting. `.claude/CLAUDE.md` is the project instruction file
and is edited conservatively: only the factual ratio (line 5) and the
quantization wording (line 10) change; no mandatory rule, naming convention, or
architectural constraint is rewritten. The `--method vq` CLI default
(`interface/cli/compress.py:27`) is left unchanged.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. `domain/model/color_codebook.py` is read only to confirm
`create_uniform_grid` is a fixed uniform grid (lines 43-57) and `quantize` is
nearest-color assignment (lines 19-29); no code, docstring, or naming is touched.

### Application Layer (`src/colors_of_meaning/application/`)

No changes.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No changes.

### Interface Layer (`src/colors_of_meaning/interface/`)

No code changes. `interface/cli/compress.py:27` (`method: str = "vq"`) is left
as-is; no flag, help string, docstring, or symbol name is modified. Only README
prose describing the mechanism is reworded.

### Shared Layer (`src/colors_of_meaning/shared/`)

No code changes. `shared/synesthetic_config.py` is read as the source of truth
(`embedding_dim = 384`, line 9; `num_bins = 4096`, line 19); it is not modified.

## Dependency Injection

No changes. No Lagom registrations are added, removed, or rewired.

## Task List

1. [ ] docs: confirm the source of truth before editing — re-verify
   `ProjectorConfig.embedding_dim = 384` (`synesthetic_config.py:9`),
   `CodebookConfig.num_bins = 4096` (`synesthetic_config.py:19`), and that
   `create_uniform_grid` (`color_codebook.py:43-57`) is a fixed uniform grid, not
   learned VQ. Confirm the ratio `12288 / 12 = 1024`.
2. [ ] docs: edit README.MD:13 — replace "extreme compression (1000x+)" with the
   precise "~1024:1 compression (384-dim float32 = 12,288 bits to a 12-bit,
   4,096-color code)".
3. [ ] docs: edit README.MD:20 — keep "12,288 bits/sentence to 12 bits
   (4,096-color palette)" and append the explicit "(~1024:1)" so figure and ratio
   agree (satisfies the README.MD:20 acceptance criterion).
4. [ ] docs: edit `.claude/CLAUDE.md:5` — replace "extreme compression (1000x+)"
   with "~1024:1 compression" consistent with the README, changing nothing else in
   that paragraph or file (conservative edit of the instruction file).
5. [ ] docs: reword the "VQ" labels at README.MD:21, README.MD:97, README.MD:183
   (and the compress prose at README.MD:193-194) to "fixed uniform Lab color
   grid" / "nearest-color quantization", and add a cross-reference to
   `007-p1-1-learned-vq-codebook` noting "VQ" terminology becomes accurate once
   learned VQ lands.
6. [ ] docs: reword `.claude/CLAUDE.md:10` from "4,096-color palette used for
   vector quantization" to describe nearest-color quantization against a fixed
   uniform Lab grid, consistent with `create_uniform_grid`.
7. [ ] docs: decide the Open Question on the "aspirational (blog 768-dim /
   2000:1) vs implemented (384-dim / ~1024:1)" note; default is to add one
   clarifying line in README.MD referencing
   https://www.qual.is/posts/colors-of-meaning.
8. [ ] docs: decide the Open Question on a config-to-docs consistency test; if
   kept simple, add a guard test (assert `ProjectorConfig().embedding_dim == 384`
   and README.MD contains "1024" and no implemented "768"/"2000:1" claim);
   otherwise defer and record the decision.
9. [ ] docs: run `tox` and confirm all eight gates pass and coverage stays 100%;
   confirm `grep -rn -i "768\|2000:1\|2000x" README.MD .claude/CLAUDE.md` shows no
   implemented-figure claim (only an explicit aspirational-blog note, if kept).

## Testing Strategy

Documentation edits do not change runtime behaviour or coverage, so the primary
gate is that `tox` stays green across all eight quality gates (flake8, black,
bandit, semgrep, xenon, mypy, pip-audit, 100% coverage). No production code is
modified, so no new behavioural tests are required by the Definition of Done.

Optionally (Task 8, an Open Question), add one cheap config-to-docs consistency
test asserting `ProjectorConfig().embedding_dim == 384` and that README.MD
contains "1024" and does not present "768"/"2000:1" as the implemented figure.
This complements the existing config-self-consistency assertion at
`tests/colors_of_meaning/shared/test_synesthetic_config.py:173` (which already
asserts `embedding_dim == 384`) by additionally pinning the docs. It must follow
house rules: one logical assertion per test, `test_should_..._when_...` naming,
and it must not flake on incidental README wording. If it cannot be kept simple and
`tox`-green, defer it and record the decision.

## Observability Plan

No changes. Documentation edits have no runtime footprint; no logging, metrics, or
tracing are added or modified.

## Risks and Mitigations

- **Risk: editing `.claude/CLAUDE.md` (the project instruction file) too broadly.**
  Mitigation: change only the factual ratio (line 5) and the quantization wording
  (line 10); leave every mandatory rule, naming convention, and architectural
  constraint untouched. The SPEC explicitly bounds these two edits.
- **Risk: a docs change unexpectedly breaks a quality gate** (e.g. a future
  doc-linked doctest or a markdown-aware check). Mitigation: run full `tox` after
  edits (Task 9); revert any wording that trips a gate.
- **Risk: reintroducing drift by softening "VQ" inconsistently** — e.g. softening
  README prose but leaving `.claude/CLAUDE.md:10` or vice versa. Mitigation: treat
  Tasks 5 and 6 as a single atomic change covering all four "VQ"/"vector
  quantization" sites; cross-reference `007-p1-1-learned-vq-codebook` uniformly.
- **Risk: stale "VQ" docs after `007-p1-1-learned-vq-codebook` lands** (the
  softened wording becomes overly cautious once learned VQ exists). Mitigation:
  the cross-reference to `007-p1-1-learned-vq-codebook` flags exactly where
  terminology should be reinstated when that step merges.
- **Risk: scope creep into code** — renaming `--method vq`
  (`interface/cli/compress.py:27`) or fixing stray `embedding_dim = 768` in
  `test_eval.py:144,218`. Mitigation: both are explicitly out of scope for this
  docs step (recorded as Open Questions); leave them and flag for follow-up unless
  the reviewer folds them in.
