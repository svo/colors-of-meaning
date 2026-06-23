# Feature: Mount Query-by-Palette API, Retire "Coconut" CRUD

## Overview
The HTTP API for a project about colors currently exposes no color functionality. `interface/api/main.py` builds a Lagom `Container()` and mounts exactly two routers: a content-free `coconut` CRUD (create/read an entity whose only field is an optional UUID `id`) and the health probes. A real, tested color endpoint already exists as `create_query_controller` in `interface/api/controller/query_controller.py` — it accepts a Lab color palette and returns the nearest documents by histogram distance — but it is **never mounted** and is therefore dead code. Its request/response Pydantic DTOs (`palette_query_dto.py`) and its use case (`QueryByPaletteUseCase`) are likewise complete but unreachable over HTTP.

This feature makes the API about colors: it **mounts the query-by-palette controller** by registering its dependency graph (`QueryByPaletteUseCase` → `CompareDocumentsUseCase` → `DistanceCalculator`, plus a `ColorCodebook` and a corpus of `ColoredDocument`) in the Lagom container in `main.py` and including its router, and it **retires the "coconut" CRUD** (controller, DTO, use cases, repository interface, in-memory implementations, and all their tests), since it carries no domain meaning. The recommendation is **mount query + retire coconut**; the alternative (delete the query controller instead, keeping the API coconut-only) is recorded in Open Questions and rejected because it would leave the project with no API surface for its actual subject.

A material runtime constraint, verified against the existing CLI wiring (`interface/cli/query.py`), shapes this work: the query path needs a populated `ColorCodebook` and a corpus of already-encoded `ColoredDocument`s, and today both are produced **offline** and loaded from pickle artifacts (`artifacts/codebooks/<name>`, `artifacts/encoded/*.pkl`). There is no in-process corpus builder. The API therefore needs an explicit strategy for "artifacts present" vs "artifacts absent at startup"; this feature specifies degraded-service behaviour rather than crashing the app.

This is roadmap step P2-1. It is engineering/honesty work (P2 tier) and is independent of the P0/P1 science steps, though it composes cleanly with 001-p0-1-lab-emd-distance (which already foreshadowed registering `DistanceCalculator` → `WassersteinDistanceCalculator` in the API container when the query controller is mounted) and 013-p2-3-real-health-checks (readiness verifying codebook/model presence — the same artifacts this endpoint needs).

## User Stories
- As an API consumer, I want to POST a Lab color palette and receive the nearest documents so that the service exposes the project's actual color-retrieval capability over HTTP.
- As a maintainer, I want the dead `create_query_controller` either wired in or deleted so that there is no unreachable production code masquerading as a feature.
- As a maintainer, I want the content-free "coconut" CRUD removed so that the API, its tests, and its coverage describe colors rather than a placeholder entity.
- As an operator, I want the query endpoint to fail gracefully with a clear status when the codebook or encoded corpus artifacts are missing so that an under-provisioned deployment degrades instead of crashing at startup.
- As a maintainer, I want the architecture tests and 100% coverage to stay green after the coconut files are deleted so that retirement does not silently drop tested code or leave orphaned imports.

## Acceptance Criteria
- [ ] Given the running app, when its OpenAPI paths are inspected, then `/query/palette` (POST) is present and the `/coconut/` and `/coconut/{id}` paths are absent.
- [ ] Given a POST to `/query/palette` with a valid `PaletteQueryRequestDTO` body and a populated corpus, when the request is handled, then the response is HTTP 200 with a `PaletteQueryResponseDTO` body (`matches: List[PaletteMatchDTO]`, `query_colors: int`) and never a plain dict.
- [ ] Given a POST to `/query/palette` with an empty `colors` list, when the request is validated, then the response is HTTP 422 (DTO `min_length=1`).
- [ ] Given the Lagom container in `main.py`, when it is built, then `QueryByPaletteUseCase`, `CompareDocumentsUseCase`, `DistanceCalculator`, `ColorCodebook`, and the corpus are resolvable and the query router is wired from container-resolved dependencies (no `coconut` types registered).
- [ ] Given the codebook or encoded-corpus artifact is missing at startup, when the app starts, then startup succeeds (no unhandled exception) and a POST to `/query/palette` returns a documented degraded response (recommended HTTP 503 with a Pydantic error DTO), not a 500 stack trace.
- [ ] Given the coconut feature is retired, when the source tree is searched, then no `src/` module imports `Coconut`, `CoconutCommandRepository`, `CoconutQueryRepository`, `Get/CreateCoconutUseCase`, the coconut DTOs, the coconut controller, or the in-memory coconut repositories.
- [ ] Given the coconut tests are removed, when `tox` runs, then all 8 gates pass and coverage is 100% with no orphaned/uncovered coconut code remaining and no test importing a deleted coconut symbol.
- [ ] Given a producer contract test for the mounted route, when it runs against the app via `TestClient`, then the live `/query/palette` response validates against `PaletteQueryResponseDTO` and the published OpenAPI schema for that path.
- [ ] Given the architecture tests, when `tox` runs, then domain remains pure, application imports domain only, and the interface wiring still satisfies the boundary rules after the edits.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)
**Retire (delete):**
- `domain/model/coconut.py` — `Coconut` entity (single optional `UUID4 id`); pure-coconut, deletable.
- `domain/repository/coconut_repository.py` — `CoconutQueryRepository`, `CoconutCommandRepository` ABCs; pure-coconut, deletable.

**Unchanged, consumed by the mounted query path (no edits):**
- `domain/model/color_codebook.py` — `ColorCodebook` (`colors: List[LabColor]`, `num_bins`, `quantize`, classmethod that builds a `bins_per_dimension**3` grid); read-only source the use case quantizes against.
- `domain/model/colored_document.py` — `ColoredDocument` (`histogram`, `num_bins`, `from_color_sequence`); the corpus element type and query-document type.
- `domain/model/lab_color.py` — `LabColor`; reconstructed from the request DTO in the controller.
- `domain/service/distance_calculator.py` — `DistanceCalculator` ABC (`compute_distance`, `metric_name`); the port `CompareDocumentsUseCase` depends on.
- `domain/repository/color_codebook_repository.py` — `ColorCodebookRepository` ABC (`save`/`load`/`exists`/`delete`); used to obtain the runtime `ColorCodebook`.

### Application Layer (`src/colors_of_meaning/application/`)
**Retire (delete):**
- `application/use_case/coconut_use_case.py` — `GetCoconutUseCase`, `CreateCoconutUseCase`; pure-coconut, deletable.

**Unchanged, now reachable over HTTP (no edits):**
- `application/use_case/query_by_palette_use_case.py` — `QueryByPaletteUseCase(compare_use_case: CompareDocumentsUseCase, codebook: ColorCodebook)`; `execute(palette, corpus_docs, k)` quantizes the palette into a normalized histogram and delegates to `find_nearest_neighbors`. Confirmed already correct; this step only wires it.
- `application/use_case/compare_documents_use_case.py` — `CompareDocumentsUseCase(distance_calculator: DistanceCalculator)`; `find_nearest_neighbors(query_doc, corpus_docs, k)` returns `List[Tuple[str, float]]`. Wired, not modified.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)
**Retire (delete):**
- `infrastructure/persistence/in_memory/in_memory_coconut_command_repository.py` — pure-coconut.
- `infrastructure/persistence/in_memory/in_memory_coconut_query_repository.py` — pure-coconut.

**Edit (mixed file):**
- `infrastructure/persistence/in_memory/shared_storage.py` — a singleton store with coconut-specific state (`coconuts: Dict[UUID, Coconut]`) and methods (`get_coconut`/`add_coconut`/`has_coconut`/`clear`). It imports `Coconut`. **Recommendation:** delete it together with the in-memory coconut repositories if nothing else uses it (verify no non-coconut consumer; the inventory shows only the two coconut repos and its own test reference it). If a non-coconut consumer is found, strip the coconut members instead. Its test must follow.

**Used by the mounted query path:**
- `infrastructure/persistence/file_color_codebook_repository.py` — `FileColorCodebookRepository(base_path="artifacts/codebooks")`; `.load(name) -> Optional[ColorCodebook]` (returns `None` when the artifact is absent). Source of the runtime codebook. No edit required, but `main.py` consumes it.
- `infrastructure/ml/wasserstein_distance_calculator.py` (and/or `jensen_shannon_distance_calculator.py`) — concrete `DistanceCalculator` bound to the port in the container. No edit here (its own constructor may take a codebook per 001-p0-1; the binding lives in `main.py`).

### Interface Layer (`src/colors_of_meaning/interface/`)
**Edit (the core of this feature):**
- `interface/api/main.py` — (1) remove all coconut wiring: imports of `Get/CreateCoconutUseCase`, `CoconutCommandRepository`/`CoconutQueryRepository`, the two in-memory coconut repos, `create_coconut_controller`; the four container registrations (lines ~34–41); and the `app.include_router(coconut_controller.router)` block (lines ~66–67). (2) Add query wiring: register `ColorCodebook` (loaded via `FileColorCodebookRepository().load(<configured name>)`, with the absent-artifact strategy), `DistanceCalculator` → concrete impl, `CompareDocumentsUseCase`, `QueryByPaletteUseCase`, and the corpus of `ColoredDocument` (loaded from the encoded-artifact path); then build the router via `create_query_controller(query_use_case, corpus_docs)` and `app.include_router(...)`. Health wiring is untouched.

**Retire (delete):**
- `interface/api/controller/coconut_controller.py` — `CoconutController` + `create_coconut_controller`.
- `interface/api/data_transfer_object/coconut_data_transfer_object.py` — request/response coconut DTOs.

**Unchanged, now mounted (no edits):**
- `interface/api/controller/query_controller.py` — `create_query_controller(query_use_case, corpus_docs) -> APIRouter`; route `POST /query/palette` returning `PaletteQueryResponseDTO`. The headline: this file stops being dead code.
- `interface/api/data_transfer_object/palette_query_dto.py` — `PaletteColorDTO`, `PaletteQueryRequestDTO`, `PaletteMatchDTO`, `PaletteQueryResponseDTO`. The request/response contract.

**CLI:** `interface/cli/main.py` has a placeholder default string `"coconuts"` (a Click message, not the domain entity). This is **out of scope** for retiring the *coconut domain feature*; changing it is cosmetic and would only churn `interface/cli/test_main.py`. Left as-is (recorded in Open Questions).

### Shared Layer
- `shared/configuration.py` / `shared/synesthetic_config.py` — no change required to retire coconut. If the codebook name and encoded-corpus path are made configurable (recommended over hardcoding), add fields here or reuse the existing CLI defaults (`codebook_4096`, `artifacts/encoded/...`). Test data strings that happen to be `"coconuts"` in `tests/.../shared/test_configuration.py` are arbitrary fixtures, not coconut-domain coupling, and may be left untouched.

## API Contracts

### Added: Query documents by color palette
- **Path:** `/query/palette`
- **Method:** `POST`
- **Tags:** `query`
- **Request body DTO:** `PaletteQueryRequestDTO` (`interface/api/data_transfer_object/palette_query_dto.py`)
  - `colors: List[PaletteColorDTO]` — required, `min_length=1`; each `PaletteColorDTO` = `{ l: float [0,100], a: float [-128,127], b: float [-128,127], weight: float > 0 (default 1.0) }`
  - `k: int` — default `5`, `ge=1`, `le=100`
- **Response 200 DTO:** `PaletteQueryResponseDTO`
  - `matches: List[PaletteMatchDTO]` where `PaletteMatchDTO = { document_id: str, distance: float }`
  - `query_colors: int` (count of palette colors echoed back)
- **422 Unprocessable Entity:** FastAPI/Pydantic validation error when `colors` is empty or a field is out of range (no custom DTO; standard validation envelope).
- **503 Service Unavailable (recommended, see Open Questions):** when the codebook or encoded corpus is unavailable at runtime; returns a Pydantic error DTO (e.g. `QueryUnavailableDTO { detail: str }`) rather than a 500. Mirrors the degraded-service posture of 013-p2-3-real-health-checks.
- **Authentication:** the existing controller adds no auth dependency (unlike coconut, which used `authentication_dependency`). Whether `/query/palette` should sit behind the existing `SecurityDependency` is an Open Question; default keeps it open (read-only retrieval).

### Removed
- `POST /coconut/` (was 201 Created, `Location` header, behind basic auth) — retired.
- `GET /coconut/{id}` (was `CoconutApiResponseDataTransferObject`, 404 on missing, behind basic auth) — retired.

## CLI Impact
No CLI command is added, removed, or changed by retiring the coconut domain. `interface/cli/query.py` already performs the analogous flow (load codebook + load encoded corpus + `QueryByPaletteUseCase`) and remains the offline counterpart of the new HTTP route — it documents the artifact contract the API now also depends on. The placeholder `"coconuts"` default string in `interface/cli/main.py` is intentionally left untouched (out of scope; see Open Questions).

## Dependency Injection
**Removed registrations (in `main.py`'s `get_container`):**
- `container[CoconutQueryRepository] = lambda: query_repo`
- `container[CoconutCommandRepository] = lambda: command_repo`
- `container[GetCoconutUseCase] = GetCoconutUseCase`
- `container[CreateCoconutUseCase] = CreateCoconutUseCase`

**Added registrations (abstract → concrete, following the existing `container[X] = lambda: instance` / `container[UseCaseClass] = UseCaseClass` pattern):**
- `ColorCodebook` → instance loaded once via `FileColorCodebookRepository(...).load(<codebook_name>)`; if `None`, fall back to the degraded path (do not register a broken instance — see Risks).
- `DistanceCalculator` → concrete `WassersteinDistanceCalculator` (constructed per 001-p0-1, e.g. from the loaded codebook) or `JensenShannonDistanceCalculator`.
- `CompareDocumentsUseCase` → `CompareDocumentsUseCase` (Lagom resolves its `DistanceCalculator`).
- `QueryByPaletteUseCase` → `QueryByPaletteUseCase` (Lagom resolves `CompareDocumentsUseCase` + `ColorCodebook`).
- The **corpus** `List[ColoredDocument]` is not a natural Lagom-resolved type; load it explicitly in `main.py` (as `query.py` does from a pickle) and pass it into `create_query_controller(query_use_case, corpus_docs)`. Document this as a deliberate non-DI construction (a loaded artifact, not a service).

`SecurityDependency`/`BasicAuthenticator` and the health registrations stay as-is. The existing `get_container`/`global_container`/`get_global_container` accessors and their tests remain (with coconut assertions removed).

## Observability
- Emit one structured `INFO` log (with `correlation-id`) at container build / app startup recording: codebook name and `num_bins` (or "absent"), corpus document count (or "absent"), and the active distance metric — so a deployment's query readiness is auditable. This is also what 013-p2-3-real-health-checks will surface on `/health/ready`.
- Log a `WARNING` (with `correlation-id`) when the codebook or corpus artifact is missing at startup, naming the expected artifact path, so the degraded mode is diagnosable.
- The query handler itself should not log per request at INFO in a hot path; a `DEBUG` entry with `correlation-id`, palette size, and `k` is acceptable. Add/keep tracing on `QueryByPaletteUseCase.execute` consistent with the project's existing use-case tracing decorators (do not introduce a new pattern).
- Removing coconut deletes its (if any) logging/tracing call sites; ensure no observability test references a deleted coconut symbol.

## Open Questions
- **Mount vs delete (the roadmap's "either/or").** Recommendation: **mount** the query controller and retire coconut. Alternative — delete `query_controller.py` + `palette_query_dto.py` + `QueryByPaletteUseCase` and keep coconut — is rejected: it leaves a colors project with no color API and discards working, tested code. If the reviewer prefers minimal scope, a middle path is "mount query now, retire coconut in a follow-up," but that leaves two API personalities; not recommended.
- **Corpus source at runtime (the real blocker).** `query.py` loads the corpus from a pre-encoded pickle (`artifacts/encoded/test_documents.pkl`) produced offline by train/encode; there is no in-process builder. Options: (a) eager-load a configured corpus artifact at startup and serve degraded 503 if absent (recommended); (b) lazy-load on first request; (c) require the artifact and fail fast at startup (rejected — crashes under-provisioned deploys). Which artifact path/name is canonical for the API, and should it be a new config field or reuse the CLI defaults?
- **Missing-artifact behaviour.** Recommended degraded response is HTTP 503 with a Pydantic error DTO. Confirm 503 vs 500 vs 404, and whether the missing-codebook case should instead be caught by `/health/ready` (013-p2-3) so the endpoint is simply not advertised as ready.
- **Distance metric for the API.** `config.distance.metric` selects `wasserstein` vs `jensen_shannon` for the CLI. Should the API bind the same configured metric (recommended) or pin one?
- **Authentication on `/query/palette`.** Coconut sat behind `SecurityDependency`; the query controller currently adds none. Keep it open (read-only) or place it behind the existing basic-auth dependency?
- **`shared_storage.py` disposition.** If verified to have no non-coconut consumer, delete it (and its test); otherwise strip only the coconut members. Confirm there is genuinely no other user.
- **`existing-coconut-example.md` scaffolder reference and CLAUDE.md/README coconut examples.** These are docs, owned by 016-p2-6-reconcile-docs. Should this step delete/update them, or only the code and tests, leaving doc reconciliation to 016? Recommendation: leave docs to 016 to avoid scope bleed; note the dependency.
- **CLI `"coconuts"` placeholder string.** Out of scope here (it is not the domain entity). Rename later if a real CLI default is decided; flagged so it is not mistaken for a missed retirement.
