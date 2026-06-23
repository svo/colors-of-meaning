# Feature: Make health checks real

## Overview

The health subsystem currently reports a liveness signal that is hardcoded
`True` and a readiness signal that only confirms the in-memory `SharedStorage`
singleton can be constructed. Neither reflects whether the application can
actually do its job, and the readiness probe ignores the artifacts the pipeline
depends on entirely.

`src/colors_of_meaning/infrastructure/system/health_checks.py` defines
`create_liveness_check`, whose inner `liveness_check` returns the literal `True`,
and `create_storage_readiness_check`, the sole readiness check registered in
`src/colors_of_meaning/infrastructure/system/health_factory.py`. As a result
`/health/ready` reports the process as ready before any codebook or projector
model exists, so an orchestrator would route traffic to an instance that cannot
encode or compare documents.

This step makes both probes meaningful while keeping the hexagonal boundary
intact. Readiness gains filesystem checks that verify the presence of the
required artifacts — the color codebook under `artifacts/codebooks/<name>.pkl`
and the projector model under `artifacts/models/projector.pth` — performed in
the infrastructure layer behind the existing `HealthChecker` domain port.
Liveness stops returning a hardcoded `True` and instead reflects real process
state (the process can execute the check and observe its own heap), so a wedged
or fatally broken process can be distinguished from a healthy one.

The probe contract is also brought into line with the house rules: the health
controller in
`src/colors_of_meaning/interface/api/controller/health_controller.py` currently
returns plain `Dict[str, Any]` from both endpoints, which violates the
"endpoints return Pydantic DTOs" rule. This step introduces explicit Pydantic
response DTOs for liveness and readiness.

No change is made to the existing `/health/live` and `/health/ready` route paths
or to the 200/503 status semantics. The wiring in
`src/colors_of_meaning/interface/api/main.py` (which resolves `HealthUseCase`
from the Lagom container and includes the health router) continues to mount the
controller unchanged in shape.

## User Stories

- As an operator, I want the readiness probe to verify that the color codebook
  and projector model artifacts are present so that traffic is only routed to an
  instance that can actually encode and compare documents.
- As an operator, I want the liveness probe to reflect real process state rather
  than a hardcoded `True` so that a wedged or fatally broken process can be
  detected and restarted.
- As an operator, I want each readiness component reported with a name, a boolean
  status, and a human-readable message so that I can see which specific
  dependency is missing when readiness fails.
- As an API consumer, I want the liveness and readiness endpoints to return a
  documented, typed response shape so that I can rely on a stable contract.
- As a maintainer, I want the health endpoints to return Pydantic DTOs rather
  than plain dictionaries so that the controller conforms to the project's
  interface-layer rules.

## Acceptance Criteria

- [ ] Given the projector model artifact `artifacts/models/projector.pth` and the
  configured codebook artifact `artifacts/codebooks/<name>.pkl` are both present,
  when `/health/ready` is called, then the response status is `200` with body
  status `ready` and every artifact component reports `status` true.
- [ ] Given the configured codebook artifact is absent, when `/health/ready` is
  called, then the response status is `503` with body status `not_ready` and the
  codebook component reports `status` false with a message identifying the
  missing codebook.
- [ ] Given the projector model artifact is absent, when `/health/ready` is
  called, then the response status is `503` with body status `not_ready` and the
  model component reports `status` false with a message identifying the missing
  model.
- [ ] Given the readiness check runs, when its component details are inspected,
  then they include both the storage component (preserved) and the artifact
  components (codebook and model) keyed by distinct component names.
- [ ] Given the liveness check runs, when its result is produced, then the value
  is derived from observed process state rather than the literal `True`, and a
  process that cannot satisfy the liveness condition reports unhealthy.
- [ ] Given any health endpoint is called, when the response is produced, then it
  is a Pydantic DTO instance (not a plain dict) with the documented fields.
- [ ] Given the readiness artifact roots are configurable, when no override is
  supplied, then the defaults `artifacts/codebooks` and `artifacts/models` are
  used, matching the CLI artifact conventions.
- [ ] Given `tox` is run, when all gates execute, then tests pass at 100%
  coverage, the domain layer imports no other layer, and the health endpoints
  return Pydantic DTOs.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

The existing port `domain/health/health_checker.py` (abstract `HealthChecker`
with `check_liveness` and `check_readiness`) and the model
`domain/health/health_status.py` (`HealthStatus` enum, `HealthResult` with
`status`, `details`, and `is_healthy`) already express everything the new checks
need: `HealthResult.details` is a `Dict[str, Dict[str, Any]]` map keyed by
component name with a `status` boolean and a `message`. No new domain abstraction
is required; readiness artifact verification is an infrastructure concern that
returns results conforming to this existing model. Domain stays pure.

### Application Layer (`src/colors_of_meaning/application/`)

No changes. `application/use_case/health_use_case.py` already delegates
`check_liveness`/`check_readiness` to the injected `HealthChecker` port and
returns the `HealthResult` unchanged. The richer details flow through unmodified.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

Primary change. In
`src/colors_of_meaning/infrastructure/system/health_checks.py`:

- Replace the hardcoded-`True` body of `create_liveness_check` so the returned
  callable reflects real process state (e.g. the process can allocate and observe
  its own runtime) rather than a literal constant, while keeping the
  `Callable[[], bool]` signature so `SystemHealthChecker.check_liveness` is
  unchanged.
- Add a codebook readiness check factory that returns a
  `Callable[[], Dict[str, Any]]` reporting a `codebook` component whose `status`
  is true only when the configured codebook artifact exists under
  `artifacts/codebooks/<name>.pkl`. Presence is determined via the existing
  `FileColorCodebookRepository.exists(name)`
  (`src/colors_of_meaning/infrastructure/persistence/file_color_codebook_repository.py`),
  reusing its `base_path` convention rather than reimplementing path logic.
- Add a model readiness check factory that returns a
  `Callable[[], Dict[str, Any]]` reporting a `model` component whose `status` is
  true only when the projector model artifact `artifacts/models/projector.pth`
  exists, using `pathlib.Path.exists` with a configurable model directory
  defaulting to `artifacts/models`.

In `src/colors_of_meaning/infrastructure/system/health_factory.py`, register the
new codebook and model readiness checks alongside the preserved storage check via
`register_readiness_check`. `create_health_checker` keeps returning the
`HealthChecker` port. No change is needed in
`src/colors_of_meaning/infrastructure/system/health_checker.py`
(`SystemHealthChecker`), which already aggregates registered checks and marks the
overall result unhealthy when any component `status` is false.

### Interface Layer (`src/colors_of_meaning/interface/`)

In `src/colors_of_meaning/interface/api/controller/health_controller.py`, change
the liveness and readiness handlers to construct and return Pydantic response
DTOs (see API Contracts) instead of plain `Dict[str, Any]`. Route paths
(`/health/live`, `/health/ready`), summaries, descriptions, and the 200/503
status behaviour are preserved. New DTO classes are added under
`src/colors_of_meaning/interface/api/data_transfer_object/` (a new
`health_dto.py`), following the existing `BaseModel`/`Field` pattern in
`coconut_data_transfer_object.py`. `src/colors_of_meaning/interface/api/main.py`
continues to resolve `HealthUseCase` from the Lagom container and include the
health router; no wiring shape change is required.

### Shared Layer

No changes required. Artifact root defaults are expressed as infrastructure
defaults matching the existing CLI conventions
(`artifacts/codebooks`, `artifacts/models`). If a single source of truth for
artifact roots is later desired, it is captured in Open Questions rather than
introduced here.

## API Contracts

Two existing endpoints retain their paths and status semantics; their response
bodies are now backed by Pydantic DTOs defined in
`src/colors_of_meaning/interface/api/data_transfer_object/health_dto.py`.

`GET /health/live` — Liveness probe.

- `200 OK` when the process is live.
- Response DTO `LivenessResponseDataTransferObject`:
  - `status: str` — `"up"` when healthy, `"down"` otherwise.
- Example: `{"status": "up"}`

`GET /health/ready` — Readiness probe.

- `200 OK` when every readiness component reports `status` true.
- `503 SERVICE UNAVAILABLE` when any readiness component reports `status` false.
- Response DTO `ReadinessResponseDataTransferObject`:
  - `status: str` — `"ready"` or `"not_ready"`.
  - `checks: Dict[str, HealthComponentDataTransferObject]` — component map keyed
    by component name (`storage`, `codebook`, `model`).
- Nested DTO `HealthComponentDataTransferObject`:
  - `status: bool`
  - `message: str`
- Example (ready):
  `{"status": "ready", "checks": {"storage": {"status": true, "message": "Storage is available"}, "codebook": {"status": true, "message": "Codebook artifact is present"}, "model": {"status": true, "message": "Model artifact is present"}}}`
- Example (not ready, missing codebook):
  `{"status": "not_ready", "checks": {"codebook": {"status": false, "message": "Codebook artifact is missing"}, ...}}`

## CLI Impact

No CLI changes. The artifact path conventions consumed by readiness already
originate in the CLI defaults
(`src/colors_of_meaning/interface/cli/train.py`,
`src/colors_of_meaning/interface/cli/encode.py`,
`src/colors_of_meaning/interface/cli/eval.py`,
`src/colors_of_meaning/interface/cli/visualize.py`): codebooks under
`artifacts/codebooks/<name>.pkl` and the projector under
`artifacts/models/projector.pth`. Readiness verifies these same locations.

## Dependency Injection

The Lagom container in `src/colors_of_meaning/interface/api/main.py` continues to
bind `HealthChecker` to the instance built by `create_health_checker` and
`HealthUseCase` to `HealthUseCase`. The new readiness check factories are
composed inside `create_health_checker`
(`src/colors_of_meaning/infrastructure/system/health_factory.py`) and registered
through `SystemHealthChecker.register_readiness_check`; the factory remains the
single composition point, so no new container binding is introduced. The
codebook check depends on `FileColorCodebookRepository`, instantiated within the
factory rather than constructed inside the check body, preserving testability and
injection of the artifact root.

## Observability

Each readiness component already carries a `status` and a human-readable
`message`, surfaced verbatim in the `checks` map of the readiness response, so a
failing dependency (storage, codebook, or model) is individually identifiable.
Liveness and readiness outcomes are logged at INFO with the existing
structured-logging facility and a `correlation-id`; a transition to unhealthy is
logged at WARNING with the failing component name. No new metric type is mandated
by this step, though a readiness gauge per component is noted in Open Questions.

## Open Questions

- Which artifacts define readiness? This spec treats the configured color
  codebook (`artifacts/codebooks/<name>.pkl`) and the projector model
  (`artifacts/models/projector.pth`) as required. Should encoded-document
  artifacts (`artifacts/encoded/...`) or a specific structured/supervised mapper
  checkpoint also gate readiness, or are codebook + projector sufficient?
- Dependency checks to include: should readiness perform a deep check that
  actually loads the sentence-transformers embedding model (and/or deserializes
  the projector and codebook) to prove they are usable, or is filesystem presence
  the intended depth for this step? A load check is heavier and may have its own
  failure modes.
- Separate liveness vs readiness endpoints: the current design keeps two distinct
  endpoints (`/health/live`, `/health/ready`). Is an aggregate `/health` endpoint
  also wanted for human inspection, and should liveness remain strictly
  process-state-only (no dependency checks) per the liveness/readiness split?
- Codebook identity: which codebook name should readiness check by default — a
  fixed conventional name, the name from `SynestheticConfig.codebook`, or any
  `.pkl` present under `artifacts/codebooks`?
- Artifact roots single source of truth: should the default roots
  (`artifacts/codebooks`, `artifacts/models`) be centralized in
  `shared/synesthetic_config.py` rather than duplicated as infrastructure
  defaults, given the CLI also hardcodes them?
