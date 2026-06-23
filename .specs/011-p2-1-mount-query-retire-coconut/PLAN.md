# Plan: Mount Query-by-Palette API, Retire "Coconut" CRUD

## Implementation Strategy
Two coordinated movements, done in one feature so the API never sits with zero color routes and never imports a deleted symbol:

1. **Mount `create_query_controller`.** In `interface/api/main.py`, register the query dependency graph in the existing Lagom `Container()` and `include_router` the query router. The graph is fully verified: `QueryByPaletteUseCase(compare_use_case: CompareDocumentsUseCase, codebook: ColorCodebook)` → `CompareDocumentsUseCase(distance_calculator: DistanceCalculator)` → a concrete `DistanceCalculator`. The `ColorCodebook` is obtained from `FileColorCodebookRepository().load(<name>)` (returns `Optional`, i.e. `None` when the artifact is absent). The **corpus** (`List[ColoredDocument]`) is not a Lagom-resolvable service — it is a loaded artifact — so it is loaded explicitly in `main.py` (mirroring `interface/cli/query.py`, which `pickle.load`s `artifacts/encoded/*.pkl`) and passed positionally into `create_query_controller(query_use_case, corpus_docs)`.

2. **Retire the coconut CRUD.** Delete the pure-coconut files across all four layers plus their tests, and edit the few mixed files (`main.py` wiring; `shared_storage.py`; the API/conftest tests that reference coconut). No coconut symbol may remain imported in `src/`.

Key decisions:
- **Degraded service over crash.** Because both required artifacts are produced offline and may be absent, the app must start regardless. Recommended: at startup, attempt to load the codebook and corpus; if either is missing, register the query path in a "degraded" mode so `POST /query/palette` returns **HTTP 503** with a small Pydantic error DTO instead of crashing at import time or 500-ing. (Confirm 503 vs deferring to `/health/ready` per 013-p2-3 — SPEC Open Question.)
- **Deterministic fallback codebook is available.** `ColorCodebook.create_uniform_grid(bins_per_dimension=16)` builds a 4096-bin codebook with no artifact, so the codebook half need never hard-fail; the corpus has no such in-process generator and is the true gating artifact.
- **Distance metric via config.** Bind `DistanceCalculator` to the concrete chosen by `config.distance.metric` (`wasserstein` / `jensen_shannon`), matching CLI behaviour; `WassersteinDistanceCalculator` is currently parameter-free but will take a codebook under 001-p0-1 — construct it the same way `eval`/`query` CLI factories do so the two steps compose.
- **PREFER EDITING.** Only `main.py` and a handful of tests are edited; the query controller, its DTOs, and its use cases are mounted unchanged. The query controller is already covered by `test_query_controller.py` (isolated, mocked); the new work is container wiring + a producer contract test against the live app.
- **Coverage is a first-class risk.** Deleting tested coconut code is fine (the code goes too), but every coconut *import* in surviving test files (`conftest.py`, `test_main.py`, `test_main_app.py`, `test_shared_storage.py`) must be removed in the same change or `tox` import-collection fails. New branches (the 503/degraded path, the artifact-absent branch in `main.py`) must be covered to keep 100%.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)
- **Delete** `domain/model/coconut.py`.
- **Delete** `domain/repository/coconut_repository.py`.
- No other domain edits. `color_codebook.py`, `colored_document.py`, `lab_color.py`, `distance_calculator.py`, `color_codebook_repository.py` are consumed unchanged. Confirm via architecture tests that domain stays pure (no infrastructure/interface import introduced).

### Application Layer (`src/colors_of_meaning/application/`)
- **Delete** `application/use_case/coconut_use_case.py`.
- No edits to `query_by_palette_use_case.py` or `compare_documents_use_case.py`; both are wired as-is. (Optional, deferred: add a use-case tracing decorator on `QueryByPaletteUseCase.execute` — but use cases are currently clean of observability, so this is minimal new startup logging in `main.py` rather than a new cross-cutting pattern; see Observability Plan.)

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
- **Delete** `infrastructure/persistence/in_memory/in_memory_coconut_command_repository.py`.
- **Delete** `infrastructure/persistence/in_memory/in_memory_coconut_query_repository.py`.
- **`infrastructure/persistence/in_memory/shared_storage.py`** — it imports `Coconut` and exposes only coconut state/methods (`coconuts` dict, `get_coconut`/`add_coconut`/`has_coconut`/`clear`). After confirming no non-coconut consumer (grep shows only the two coconut repos + its own test), **delete it**; if a non-coconut consumer is found, strip the coconut members instead.
- No edits to `file_color_codebook_repository.py` or the distance calculators; they are consumed by the new wiring.

### Interface Layer (`src/colors_of_meaning/interface/`)
- **Edit `interface/api/main.py`** (central change):
  - Remove coconut imports (`Get/CreateCoconutUseCase`, `CoconutCommandRepository`/`CoconutQueryRepository`, the two in-memory coconut repos, `create_coconut_controller`).
  - Remove the four coconut container registrations and the `coconut_controller`/`include_router` block.
  - In `get_container`, register: `ColorCodebook` (from `FileColorCodebookRepository().load(<name>)`, falling back per the degraded strategy), `DistanceCalculator` → concrete-by-config, `CompareDocumentsUseCase`, `QueryByPaletteUseCase`.
  - Load the corpus artifact explicitly (try/except around the pickle load; empty/degraded on failure) and build `create_query_controller(global_container[QueryByPaletteUseCase], corpus_docs)`, then `app.include_router(...)`.
  - Keep `get_container`/`global_container`/`get_global_container`, health wiring, and `main`/`run` intact.
- **Delete** `interface/api/controller/coconut_controller.py`.
- **Delete** `interface/api/data_transfer_object/coconut_data_transfer_object.py`.
- **(If the 503 degraded contract is adopted)** add a tiny `QueryUnavailableDTO(BaseModel)` (e.g. `detail: str`) in `interface/api/data_transfer_object/palette_query_dto.py` (edit, not new file) so the degraded response is a Pydantic DTO, never a dict.
- No edits to `query_controller.py` or `palette_query_dto.py` beyond the optional error DTO. No CLI edits (the `"coconuts"` placeholder string is out of scope).

### Shared Layer (`src/colors_of_meaning/shared/`)
- No change required for retirement. If the codebook name / corpus artifact path are made configurable (recommended over hardcoding in `main.py`), add fields to `shared/synesthetic_config.py` or `shared/configuration.py`, reusing the CLI defaults (`codebook_4096`, `artifacts/encoded/...`). Arbitrary `"coconuts"` test-data strings in `tests/.../shared/test_configuration.py` are not coconut-domain coupling and need not change.

## Dependency Injection
Following the verified existing patterns in `main.py` (`container[Abstract] = lambda: instance`, `container[UseCaseClass] = UseCaseClass`, retrieve via `container[X]`):

Removed: `CoconutQueryRepository`, `CoconutCommandRepository`, `GetCoconutUseCase`, `CreateCoconutUseCase`.

Added:
- `container[ColorCodebook] = lambda: codebook` where `codebook = FileColorCodebookRepository().load(name)`; if `None`, use the degraded path (do not register a broken value used by a live route).
- `container[DistanceCalculator] = lambda: <WassersteinDistanceCalculator(...) | JensenShannonDistanceCalculator(...)>` selected by `config.distance.metric`.
- `container[CompareDocumentsUseCase] = CompareDocumentsUseCase` (Lagom resolves `DistanceCalculator`).
- `container[QueryByPaletteUseCase] = QueryByPaletteUseCase` (Lagom resolves `CompareDocumentsUseCase` + `ColorCodebook`).
- Corpus `List[ColoredDocument]` is **not** registered in Lagom (it is a loaded artifact, not a service); it is constructed in `main.py` and injected into `create_query_controller` positionally — documented as a deliberate non-DI construction.

Security/health registrations unchanged.

## Task List
1. [ ] domain: delete `domain/model/coconut.py` and `domain/repository/coconut_repository.py`. (AC: no `src/` import of coconut domain types.)
2. [ ] application: delete `application/use_case/coconut_use_case.py`. (AC: no `src/` import of coconut use cases.)
3. [ ] infrastructure: delete `in_memory_coconut_command_repository.py` and `in_memory_coconut_query_repository.py`; after grep-confirming no non-coconut consumer, delete `shared_storage.py`. (AC: no `src/` import of coconut repos/storage.)
4. [ ] interface: delete `coconut_controller.py` and `coconut_data_transfer_object.py`. (AC: `/coconut/*` paths absent from OpenAPI.)
5. [ ] interface: edit `interface/api/main.py` — strip all coconut wiring (imports, four registrations, controller/router block) while keeping `get_container`/`global_container`/`get_global_container`, health wiring, `main`/`run`. (AC: container builds with no coconut types; coconut paths absent.)
6. [ ] interface: in `main.py`, register `ColorCodebook` (via `FileColorCodebookRepository().load(name)` with `create_uniform_grid` / degraded fallback), `DistanceCalculator` (by `config.distance.metric`), `CompareDocumentsUseCase`, `QueryByPaletteUseCase`. (AC: query graph resolvable from container.)
7. [ ] interface: in `main.py`, load the corpus artifact (try/except → degraded/empty on absence), build `create_query_controller(container[QueryByPaletteUseCase], corpus_docs)`, and `app.include_router(...)`. (AC: `/query/palette` POST present in OpenAPI; degraded behaviour when artifact absent.)
8. [ ] interface: if adopting the degraded contract, add `QueryUnavailableDTO` to `palette_query_dto.py` and return HTTP 503 with it when codebook/corpus is unavailable (never a dict). (AC: missing-artifact → 503 + Pydantic DTO, not 500.)
9. [ ] shared (optional): add codebook-name / corpus-path config fields reusing CLI defaults, instead of hardcoding in `main.py`. (Supports the artifact-source Open Question.)
10. [ ] tests: delete the 8 pure-coconut test files (domain model, both repo contract tests, use-case test, both in-memory repo tests, controller test, controller benchmark, DTO test) and `test_shared_storage.py`. (AC: no test imports a deleted coconut symbol.)
11. [ ] tests: edit `conftest.py` to remove the coconut fixtures; edit `interface/api/test_main.py` to drop coconut container/controller setup; edit `interface/api/test_main_app.py` to remove the `/coconut/` and `/coconut/{id}` route assertions (replace with `/query/palette` presence). (AC: architecture/import collection green; coconut paths asserted absent, query path asserted present.)
12. [ ] tests: add wiring tests for the query registrations in `main.py` (each type resolvable from the container; corpus loaded; degraded branch covered). (AC: container resolves query graph; 100% coverage incl. degraded branch.)
13. [ ] tests: add the **producer CDCT** for `POST /query/palette` against the live `app` via `TestClient` — 200 + `PaletteQueryResponseDTO` shape on a populated corpus, 422 on empty `colors`, and (if adopted) 503 + `QueryUnavailableDTO` on absent artifact. (AC: live response validates against DTO + OpenAPI.)
14. [ ] tests: run `tox` (8 gates) and `tox -e format`; confirm 100% coverage and that `grep -r coconut src/` is empty. (AC: tox green, coverage 100%.)

## Testing Strategy
- **Frameworks per house rules:** `assertpy` (`assert_that`) for entity/DTO/app-config assertions (the existing API tests use `assert_that`); plain `assert` / `pytest.raises` and `TestClient` for the route/integration tests (matching `test_query_controller.py`). **One logical assertion per test**; related asserts on one HTTP response (status + body shape) may group. Names follow `test_should_<behaviour>_when_<condition>`.
- **Reuse, don't duplicate:** keep `test_query_controller.py` (isolated controller behaviour with a mock use case and a synthetic corpus) as-is — it already covers 200, matches, query-colors count, 422-empty, and use-case invocation. The new tests cover *wiring and the live app*, which it does not.
- **Producer CDCT (required for the route we now provide):** in `tests/colors_of_meaning/interface/api/test_main_app.py` (or a sibling), drive the real `app` with `TestClient`:
  - `test_should_expose_query_palette_route_when_app_is_built` — assert `/query/palette` POST in `app.openapi()` paths.
  - `test_should_return_palette_query_response_dto_when_palette_posted` — POST a valid palette against a populated test corpus; assert the JSON validates as `PaletteQueryResponseDTO` (`matches`, `query_colors`) — the producer contract.
  - `test_should_return_422_when_colors_is_empty` — empty `colors` → 422.
  - `test_should_return_503_when_query_artifacts_are_unavailable` — (if degraded contract adopted) absent codebook/corpus → 503 with `QueryUnavailableDTO`.
- **Coconut-absence assertions (replacing the deleted route tests):**
  - `test_should_not_expose_create_coconut_route_when_app_is_built` and `test_should_not_expose_get_coconut_route_when_app_is_built` — assert `/coconut/` and `/coconut/{id}` absent from OpenAPI paths (the inverse of the current `test_should_have_*_coconut_route` tests being removed).
- **Container wiring tests:** `test_should_resolve_query_by_palette_use_case_when_container_built`, `test_should_resolve_compare_documents_use_case_when_container_built`, `test_should_resolve_color_codebook_when_container_built`, `test_should_resolve_distance_calculator_when_container_built` — each asserts `get_container()[X]` is not `None` / is the expected type. Plus `test_should_load_corpus_when_artifact_present` and `test_should_degrade_when_corpus_artifact_absent` (patch the loader) to cover both branches.
- **Architecture / boundary tests (`pytest-archon` in `test_architecture.py` / `test_synesthetic_architecture.py`):** must stay green — domain pure, application→domain only, infrastructure implements ports, interface wires. Confirm no rule referenced coconut specifically; if one enumerated coconut modules, update it.
- **Artifact independence:** tests must not depend on real `artifacts/` files or network. Build a small in-memory `ColorCodebook` (e.g. `create_uniform_grid(2)` → 8 bins) and a tiny synthetic corpus, and patch the codebook/corpus loaders in `main.py` so the live-app tests are deterministic and fast.
- **Verification:** `tox` for all 8 gates (never `pytest` alone); `tox -- tests/colors_of_meaning/interface/api/` for fast loops; `tox -e format` before completion; 100% coverage including the degraded/absent branches; final `grep -rn -i coconut src/` returns nothing.

## Observability Plan
- The codebase's use cases and distance calculators are currently free of observability decorators; this step does **not** introduce a new cross-cutting decorator pattern. Instead, add minimal **startup** logging in `main.py`:
  - one structured `INFO` (with `correlation-id`) recording codebook name + `num_bins` (or "absent"), corpus document count (or "absent"), and the active distance metric — making query readiness auditable and feeding 013-p2-3's readiness check;
  - a `WARNING` (with `correlation-id`) naming the expected artifact path when codebook or corpus is missing at startup (degraded mode is diagnosable).
- The query handler may emit a `DEBUG` line (correlation-id, palette size, `k`); no per-request INFO logging in the hot path; no new metrics/tracing mandated.
- Deleting coconut removes its call sites; ensure no observability/logging test references a deleted coconut symbol.

## Risks and Mitigations
- **Risk:** corpus artifact absent at API startup → `pickle.load` raises and the module-level wiring crashes the whole app at import. → **Mitigation:** load inside try/except in `main.py`, fall back to a degraded controller returning 503; cover both branches by patching the loader (tasks 7, 8, 12, 13).
- **Risk:** deleting coconut tests/code leaves surviving test files importing deleted symbols, so `tox` fails at collection (looks like a coverage/gate failure, not an import error). → **Mitigation:** delete and edit coconut imports in the same change (tasks 10–11); run `grep -rn -i coconut tests/` before `tox`.
- **Risk:** `shared_storage.py` (or another file) has a hidden non-coconut consumer, so deleting it breaks an unrelated path. → **Mitigation:** grep for importers before deleting (task 3); strip coconut members instead of deleting if a consumer exists.
- **Risk:** new branches (503/degraded, artifact-absent) drop coverage below 100%. → **Mitigation:** explicit tests for both present and absent artifact paths (task 12) and for the 503 response (task 13).
- **Risk:** mounting changes the published OpenAPI (`build/openapi.json` regenerated by `test_main_app.py`), surprising consumers of the old coconut contract. → **Mitigation:** this is the intended contract change; the producer CDCT pins the new `/query/palette` schema and the removed coconut paths are asserted absent.
- **Risk:** `WassersteinDistanceCalculator` is parameter-free today but gains a `ColorCodebook` constructor arg under 001-p0-1, so wiring written now could break when that lands. → **Mitigation:** construct the calculator exactly as the `eval`/`query` CLI factories do (single source of construction convention) so both steps compose; if 001 has not landed, the parameter-free constructor still works.
- **Risk:** binding `DistanceCalculator` by `config.distance.metric` introduces a config dependency in `main.py` that may be unset. → **Mitigation:** default to `wasserstein` when unset; cover the selection in a wiring test.
- **Risk:** scope bleed into docs (CLAUDE.md, README, scaffolder `existing-coconut-example.md` all reference coconut). → **Mitigation:** treat code+tests here and leave doc reconciliation to 016-p2-6-reconcile-docs; note the cross-reference (SPEC Open Question) so reviewers expect the doc follow-up.
- **Risk:** removing basic-auth (coconut had it) while `/query/palette` adds none could be read as a security regression. → **Mitigation:** decide auth posture explicitly (SPEC Open Question); default open is acceptable for read-only retrieval, but record the decision.
