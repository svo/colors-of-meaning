# Plan: Unify the two config systems so the science config reaches the API runtime

## Implementation Strategy

Compose the science config into the API runtime rather than merging the two schemas.
Both modules stay in `src/colors_of_meaning/shared/` and stay importable by every layer.
The work adds a thin runtime-context component in `shared/` that resolves an
experiment-config path plus the artifact paths the API needs (model, codebook, encoded
corpus), loads `SynestheticConfig` via the unchanged `SynestheticConfig.from_yaml`, and
exposes a typed runtime context with a single artifact-path accessor. That context and
the loaded `SynestheticConfig` are then registered by type in the Lagom `Container()`
built in `interface/api/main.py::get_container()`, alongside the existing registrations.

The load-bearing constraint is back-compat: the five CLI commands
(`train`, `eval`, `encode`, `visualize`, `compare`) must keep calling
`SynestheticConfig.from_yaml(args.config)` with no flag or schema change, and the
existing `configs/*.yaml` must parse identically. The chosen design touches the CLI not
at all.

Sequencing:

1. Add the runtime-context component to `shared/` (prefer extending
   `shared/configuration.py`) with explicit-override support for tests.
2. Wire it into `interface/api/main.py::get_container()` so the container resolves the
   runtime context and `SynestheticConfig` by type.
3. Coordinate the artifact-path accessor with `013-p2-3-real-health-checks` (one source
   of truth) and leave route mounting to `011-p2-1-mount-query-retire-coconut`.
4. Add tests (shared component, container wiring, CLI back-compat, architecture).
5. Run `tox`; confirm eight gates green and 100% coverage.

Because step 2 makes `get_container()` load real artifacts (codebook, model, encoded
corpus), also collapse the existing **duplicated import-time** container construction in
`main.py` — the module-level `global_container = get_container()` plus the second build
inside `create_app()` — down to a single construction, so artifacts load once and the
`012-p2-2-hash-credentials` fail-closed credential warning is emitted once rather than
twice at import. Remove the redundant eager `global_container = get_container()` and
expose it lazily through `get_global_container()`; keep `create_app()` building its own
container (the query-API tests patch `_load_query_codebook` / `_load_corpus` around it),
and keep `app = create_app()` at module scope so `uvicorn` can import `...main:app`. This
is pre-existing wiring debt surfaced by the P2-2 adversarial review; P2-4 owns it because
P2-4 is what makes the duplicate build expensive.

The alternative (keep two systems, add a documented bridge with an optional
`experiment_config_path`) is carried as an Open Question in SPEC.md and is not
implemented unless the unification recommendation is rejected.

## Layer Changes

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. Config stays in `shared/`; no domain entity, port, or service is added.

### Application Layer (`src/colors_of_meaning/application/`)

No changes. Use cases keep receiving collaborators by constructor injection and never
import config-loading machinery. Any experiment parameter a use case needs is passed in
as a plain value by the interface wiring.

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new adapters. The runtime context composes existing loaders only:
`FileColorCodebookRepository` (codebook), `PyTorchColorMapper.load_weights` (model), and
the pickled encoded corpus already read by `interface/cli/query.py`.
`infrastructure/security/basic_authentication.py` keeps reading credentials from the
runtime settings unchanged (owned by `012-p2-2-hash-credentials`).

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/api/main.py::get_container()` gains registration of the runtime context and
the loaded `SynestheticConfig`, alongside the current coconut, auth, and health
registrations. The CLI entry points (`train.py`, `eval.py`, `encode.py`, `visualize.py`,
`compare.py`) are unchanged. Mounting `create_query_controller` against the new context
is deferred to `011-p2-1-mount-query-retire-coconut`. As part of this step, also remove
the redundant module-level `global_container = get_container()` (build it lazily via
`get_global_container()`) so the heavier container is constructed once at import rather
than twice; `create_app()` and the module-level `app = create_app()` stay so the app
remains importable for `uvicorn` and rebuildable for tests.

### Shared Layer (`src/colors_of_meaning/shared/`)

`shared/configuration.py` and `shared/synesthetic_config.py` both remain and both stay
importable by all layers. Add the runtime-context component here — preferably extending
`shared/configuration.py` (e.g. a runtime context that holds the loaded
`SynestheticConfig` and resolved artifact paths, plus a factory mirroring
`get_application_setting_provider`) over creating a new module, unless cohesion clearly
argues otherwise. The two existing schemas are not collapsed into one class: science
config stays nested dataclasses with YAML round-trip; runtime config stays Pydantic
`BaseSettings`.

## Dependency Injection

Register the runtime context and the `SynestheticConfig` by type in the Lagom
`Container()` in `interface/api/main.py::get_container()`, next to
`CoconutQueryRepository`, `CoconutCommandRepository`, the use cases,
`BasicAuthenticator`, `SecurityDependency`, and `HealthChecker`. Resolution stays
type-based. Tests inject an in-memory `SynestheticConfig` (constructed directly) so the
container builds without reading `configs/` from disk. The CLI path stays DI-free and
unchanged.

## Task List

1. [ ] domain: confirm no domain change is required and record it (config is a
   `shared/` concern; domain stays pure).
2. [ ] application: confirm no application change is required and record it (use cases
   receive experiment values by injection, never import config loaders).
3. [ ] shared: add a runtime-context component in `shared/` (prefer extending
   `shared/configuration.py`) that holds a loaded `SynestheticConfig` and resolved
   artifact paths (model, codebook, encoded corpus), exposing a single artifact-path
   accessor.
4. [ ] shared: load `SynestheticConfig` through the existing
   `SynestheticConfig.from_yaml`, accepting an explicit in-memory `SynestheticConfig`
   override so tests need no disk YAML.
5. [ ] shared: resolve the experiment-config path from the existing
   `ApplicationSettings` env/`application.properties` mechanism (respecting the
   `_apply_property` precedence), and fail fast with a clear, path-naming error when the
   config or its artifacts are missing.
6. [ ] infrastructure: confirm the runtime context composes only existing loaders
   (`FileColorCodebookRepository`, `PyTorchColorMapper.load_weights`, pickled corpus)
   and adds no new adapter; leave `basic_authentication.py` untouched.
7. [ ] interface: register the runtime context and the loaded `SynestheticConfig` by
   type in `interface/api/main.py::get_container()`, alongside the existing
   registrations.
8. [ ] interface: verify the CLI entry points (`train.py`, `eval.py`, `encode.py`,
   `visualize.py`, `compare.py`) remain unchanged and still load via
   `SynestheticConfig.from_yaml(args.config)`.
9. [ ] interface: add startup-time structured logging (with `correlation-id`) of the
   resolved experiment-config and artifact paths at INFO and load failures at ERROR,
   never logging credential values.
10. [ ] tests: add shared-component tests (config resolution, override, artifact-path
    accessor, fail-fast on missing path).
11. [ ] tests: add a container-wiring test that the Lagom `Container()` resolves
    `SynestheticConfig` and the runtime context by type.
12. [ ] tests: add CLI back-compat tests that each `configs/*.yaml` parses into the same
    `SynestheticConfig` and that CLI config loading is unchanged.
13. [ ] tests: add a pytest-archon architecture test that `shared/` is importable by all
    layers and that neither config module leaks into `domain/`.
14. [ ] interface: coordinate the single artifact-path accessor with
    `013-p2-3-real-health-checks` and confirm `011-p2-1-mount-query-retire-coconut` is
    the owner of any route mounting.
15. [ ] interface: collapse the duplicated import-time container construction in
    `main.py` — remove the eager `global_container = get_container()`, expose it lazily
    via `get_global_container()`, and ensure the heavier container (and the
    `012-p2-2-hash-credentials` fail-closed warning) is built/emitted once; keep
    `create_app()` rebuildable for tests and `app = create_app()` importable for
    `uvicorn` (surfaced by the P2-2 adversarial review).
16. [ ] tests: assert importing `interface.api.main` constructs the container once and
    emits the fail-closed credential warning at most once (no double import-time build);
    assert `get_global_container()` still returns a usable container and `create_app()`
    still builds a patchable one.
17. [ ] tests: run `tox` and confirm all eight gates pass and coverage is 100%.

## Testing Strategy

All tests follow house rules: one logical assertion each (related assertions on the
same result allowed for ML/structured outputs), `test_should_..._when_...` naming,
`assertpy` (`assert_that`) for config-entity assertions and plain `assert`/`pytest.raises`
for any ML/domain edges, no comments. Tests mirror under
`tests/colors_of_meaning/<layer>/...`. No network and no fixed-path disk I/O in unit
tests — inject an in-memory `SynestheticConfig` override.

- Shared component (mirrors `shared/`):
  - `test_should_load_synesthetic_config_when_experiment_path_is_configured`
  - `test_should_use_injected_config_when_override_is_provided` (no disk read)
  - `test_should_expose_resolved_artifact_paths_when_context_is_built`
  - `test_should_raise_when_experiment_config_path_is_missing` (`pytest.raises`)
  - `test_should_preserve_runtime_settings_when_context_is_built` (host/reload still
    resolve, credentials untouched)
- Container wiring (mirrors `interface/api/`):
  - `test_should_resolve_synesthetic_config_from_container_when_built`
  - `test_should_resolve_runtime_context_from_container_when_built`
  - `test_should_build_container_once_when_module_is_imported`
  - `test_should_emit_fail_closed_warning_at_most_once_when_module_is_imported`
- CLI back-compat (mirrors `interface/cli/` and `shared/`):
  - `test_should_parse_base_config_unchanged_when_loaded_from_yaml`
  - `test_should_parse_structured_config_unchanged_when_loaded_from_yaml`
  - `test_should_parse_supervised_config_unchanged_when_loaded_from_yaml`
- Architecture (pytest-archon):
  - `test_should_allow_all_layers_to_import_shared`
  - `test_should_not_import_configuration_module_in_domain_layer`
- Final gate: `tox` (never `pytest` alone) — flake8, black, bandit, semgrep, pip-audit,
  radon, xenon, mypy — with 100% coverage.

## Observability Plan

Use the existing structured logger with a `correlation-id`. At container build time, log
the resolved experiment-config path and artifact paths (model, codebook, encoded corpus)
at INFO and any load failure at ERROR with the offending path. Never log credential
values from the runtime settings. No new metrics or tracing spans are required; if added
later, follow existing conventions.

## Risks and Mitigations

- Risk: the architectural change breaks CLI YAML loading. Mitigation: the chosen design
  does not touch the CLI; back-compat tests assert each `configs/*.yaml` parses
  identically (Tasks 8, 12).
- Risk: migration disturbs the runtime settings (host/reload/credentials). Mitigation:
  compose alongside `ApplicationSettings` without altering its fields; assert runtime
  settings still resolve (Task 10); leave credential handling to
  `012-p2-2-hash-credentials`.
- Risk: artifact paths get defined in two places and drift from
  `013-p2-3-real-health-checks`. Mitigation: expose one artifact-path accessor and have
  health checks read it (Tasks 3, 14).
- Risk: duplication/conflict with `011-p2-1-mount-query-retire-coconut` over mounting
  the query controller. Mitigation: P2-4 only makes the context resolvable; mounting is
  explicitly out of scope and owned by `011` (Tasks 7, 14, and the SPEC ordering Open
  Question).
- Risk: tests depend on disk YAML or large artifacts, slowing the suite and breaking
  isolation. Mitigation: inject an in-memory `SynestheticConfig` override so the
  container builds without reading `configs/` (Tasks 4, 11).
- Risk: silent partial startup when the experiment config is set but artifacts are
  missing. Mitigation: fail fast at build time with a path-naming error
  (Task 5); the degraded-mode alternative is an Open Question for
  `013-p2-3-real-health-checks`.
- Risk: scope creep into merging the two schemas into one class. Mitigation: keep both
  modules and both schema styles; deliver only a composing runtime context (Strategy,
  Shared Layer).
- Risk: leaking credentials into logs. Mitigation: log only paths, never credential
  values (Task 9, Observability Plan).
- Risk: the heavier `get_container()` is built twice at import (pre-existing duplicate
  build), loading artifacts twice and emitting the `012-p2-2-hash-credentials` fail-closed
  warning twice. Mitigation: remove the eager `global_container = get_container()` and
  build lazily via `get_global_container()` so construction happens once; keep
  `create_app()` rebuildable for tests and `app` importable for `uvicorn` (Tasks 15, 16).
- Risk: de-eagering construction breaks the query-API tests that patch
  `_load_query_codebook` / `_load_corpus` around `create_app()`, or breaks `uvicorn`'s
  `...main:app` import. Mitigation: keep `create_app()` building its own container and
  keep `app = create_app()` at module scope; only the redundant `global_container` eager
  build is removed (Tasks 15, 16).
