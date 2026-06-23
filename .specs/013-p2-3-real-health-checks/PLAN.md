# Plan: real-health-checks

## Implementation Strategy

Make the health probes meaningful without disturbing the hexagonal boundary or
the existing route contract. The domain port
(`src/colors_of_meaning/domain/health/health_checker.py`) and model
(`src/colors_of_meaning/domain/health/health_status.py`) already express the
component-keyed `details` map this work needs, so all behavioural change lands in
infrastructure (new readiness checks plus a real liveness signal) and interface
(Pydantic response DTOs). `SystemHealthChecker`
(`src/colors_of_meaning/infrastructure/system/health_checker.py`) already
aggregates registered checks and fails the overall result when any component
`status` is false, so the new readiness checks plug in through the existing
`register_readiness_check` path with no aggregator change.

Work proceeds test-first in layer order. Readiness artifact presence reuses
`FileColorCodebookRepository.exists`
(`src/colors_of_meaning/infrastructure/persistence/file_color_codebook_repository.py`)
for codebooks and a `pathlib.Path.exists` check for the projector model, with
configurable roots defaulting to `artifacts/codebooks` and `artifacts/models` to
match the CLI conventions. Liveness stops returning the literal `True`. The
controller is migrated to typed DTOs to satisfy the interface-layer rule. All
quality gates are verified with `tox`.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. `HealthChecker` port and `HealthResult`/`HealthStatus` model already
support a component-keyed `details` map with `status` and `message`. Domain
remains pure (no new imports of other layers).

### Application Layer (`src/colors_of_meaning/application/`)

No changes. `application/use_case/health_use_case.py` already delegates to the
`HealthChecker` port and returns `HealthResult` unchanged; richer details pass
through transparently.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

Edit `src/colors_of_meaning/infrastructure/system/health_checks.py`:

- Rework `create_liveness_check` so the returned `Callable[[], bool]` derives its
  value from observed process state rather than the constant `True`, keeping the
  signature stable.
- Add `create_codebook_readiness_check(base_path="artifacts/codebooks", name=...)`
  returning a `Callable[[], Dict[str, Any]]` that reports a `codebook` component;
  `status` is true only when `FileColorCodebookRepository(base_path).exists(name)`
  is true, with a present/missing `message`.
- Add `create_model_readiness_check(models_path="artifacts/models", model_file="projector.pth")`
  returning a `Callable[[], Dict[str, Any]]` that reports a `model` component;
  `status` is true only when the model file exists under the configured directory.

Edit `src/colors_of_meaning/infrastructure/system/health_factory.py` to register
the new codebook and model readiness checks alongside the preserved storage check
within `create_health_checker`, still returning the `HealthChecker` port. No edit
to `src/colors_of_meaning/infrastructure/system/health_checker.py`.

### Interface Layer (`src/colors_of_meaning/interface/`)

Create `src/colors_of_meaning/interface/api/data_transfer_object/health_dto.py`
with Pydantic models: `HealthComponentDataTransferObject` (`status: bool`,
`message: str`), `LivenessResponseDataTransferObject` (`status: str`), and
`ReadinessResponseDataTransferObject`
(`status: str`, `checks: Dict[str, HealthComponentDataTransferObject]`),
following the `BaseModel`/`Field` style of
`src/colors_of_meaning/interface/api/data_transfer_object/coconut_data_transfer_object.py`.

Edit `src/colors_of_meaning/interface/api/controller/health_controller.py` so the
liveness and readiness handlers build and return these DTOs instead of plain
`Dict[str, Any]`, preserving the `/health/live` and `/health/ready` paths and the
200/503 semantics. No shape change required in
`src/colors_of_meaning/interface/api/main.py`.

### Shared Layer (`src/colors_of_meaning/shared/`)

No changes. Artifact root defaults remain infrastructure defaults matching CLI
conventions; centralization is deferred (see SPEC Open Questions).

## Dependency Injection

`create_health_checker`
(`src/colors_of_meaning/infrastructure/system/health_factory.py`) remains the
single composition point: it instantiates the readiness check factories
(including the `FileColorCodebookRepository` the codebook check uses) and
registers them via `SystemHealthChecker.register_readiness_check`. The Lagom
container in `src/colors_of_meaning/interface/api/main.py` keeps binding
`HealthChecker` to the factory result and `HealthUseCase` to `HealthUseCase`. No
new container binding is added. Artifact roots are passed as factory arguments,
keeping checks injectable and testable without touching the filesystem of the
real `artifacts/` tree.

## Task List

1. [ ] domain: confirm `HealthResult.details` supports component-keyed
   `status`/`message` entries; no code change, record as a no-op in the plan.
2. [ ] application: confirm `HealthUseCase` passes `HealthResult` through
   unchanged; no code change.
3. [ ] infrastructure: in `health_checks.py`, replace the hardcoded `True` in
   `create_liveness_check` with a real process-state signal (stable signature).
4. [ ] infrastructure: in `health_checks.py`, add `create_codebook_readiness_check`
   reporting a `codebook` component via `FileColorCodebookRepository.exists`.
5. [ ] infrastructure: in `health_checks.py`, add `create_model_readiness_check`
   reporting a `model` component via `Path.exists` under the models root.
6. [ ] infrastructure: in `health_factory.py`, register the codebook and model
   readiness checks alongside the preserved storage check.
7. [ ] infrastructure: add structured logging (correlation-id) for liveness and
   readiness outcomes, WARNING on transition to unhealthy with the component name.
8. [ ] interface: create
   `interface/api/data_transfer_object/health_dto.py` with
   `HealthComponentDataTransferObject`, `LivenessResponseDataTransferObject`,
   `ReadinessResponseDataTransferObject`.
9. [ ] interface: edit `health_controller.py` to return the new DTOs from the
   liveness and readiness handlers, preserving paths and 200/503 semantics.
10. [ ] tests: infrastructure — liveness reflects process state, not literal
    `True`; readiness true/false for codebook present/absent; readiness
    true/false for model present/absent; component messages for each case;
    factory registers storage + codebook + model checks.
11. [ ] tests: interface — readiness returns 200/`ready` and 503/`not_ready`;
    `checks` map carries typed components; liveness returns `up`/`down`;
    responses are DTO instances with documented fields.
12. [ ] tests: architecture — extend/keep pytest-archon rules so the new DTO
    module imports no domain model and the controller still depends on the
    application use case.
13. [ ] verify: run `tox` (all 8 gates) and confirm 100% coverage; format with
    `tox -e format`.

## Testing Strategy

One logical assertion per test; ML/numerical groupings do not apply here, so each
test asserts a single behaviour. Use `assertpy` (`assert_that`) for entity and
DTO/controller assertions, consistent with the existing
`tests/colors_of_meaning/infrastructure/system/test_health_checks.py`,
`test_health_factory.py`, and
`tests/colors_of_meaning/interface/api/controller/test_health_controller.py`.
Plain `assert`/`pytest.raises` is reserved for any domain/ML-style checks; none
are expected here. Names follow `test_should_<behaviour>_when_<condition>`.

Filesystem presence is exercised with `tmp_path` (pointing the artifact roots at
a temporary directory) and `unittest.mock.patch` for failure injection, mirroring
the existing storage-check tests; no real `artifacts/` access and no network.
Readiness tests cover: codebook present, codebook absent, model present, model
absent, and both-present aggregate ready. Controller tests use FastAPI
`TestClient` with a mocked `HealthUseCase` (as the current suite does) to assert
status codes, body status strings, the typed `checks` map, and that responses are
Pydantic DTO instances. Architecture is enforced with `pytest_archon` `archrule`
in `tests/colors_of_meaning/test_architecture.py`
(domain independence; DTO module not importing `domain.model`; controller
importing the application use case). Final verification is `tox` only (never
`pytest` alone) at 100% coverage.

## Observability Plan

Emit structured logs with a `correlation-id` for each liveness and readiness
evaluation at INFO, and at WARNING when overall health transitions to unhealthy,
including the failing component name (`storage`, `codebook`, or `model`). Each
readiness component's `message` is surfaced in the `checks` response so the
specific missing artifact is visible to operators. A per-component readiness gauge
is left as an Open Question rather than introduced in this step.

## Risks and Mitigations

- Risk: real-artifact readiness checks accidentally touch or create the live
  `artifacts/` tree during tests. Mitigation: inject artifact roots and point
  them at `tmp_path`; never construct repositories against default paths in tests.
- Risk: `FileColorCodebookRepository.__init__` calls `mkdir(parents=True,
  exist_ok=True)`, so merely constructing it creates the directory. Mitigation:
  treat directory existence as irrelevant to readiness and assert on
  `.exists(name)` (file presence) only; in tests use `tmp_path` so any created
  directory is disposable.
- Risk: changing the controller return type breaks the existing
  `test_health_controller.py` JSON-shape expectations. Mitigation: keep the JSON
  body identical (same keys/values) so DTO serialization is wire-compatible;
  update assertions only where they newly verify DTO typing.
- Risk: making liveness depend on process state could flap or reduce coverage of
  the negative branch. Mitigation: keep the liveness condition deterministic and
  cheap, and unit-test the unhealthy branch via mocking so both branches are
  covered to 100%.
- Risk: a deep load check (embedding model / deserialization) is heavier and has
  its own failure modes. Mitigation: scope this step to filesystem presence and
  defer load-depth to the SPEC Open Question.
- Risk: route or DI shape drift in `main.py`. Mitigation: leave container
  bindings and router inclusion unchanged; confine edits to the controller body
  and the new DTO module.
