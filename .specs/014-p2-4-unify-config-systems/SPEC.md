# Feature: Unify the two config systems so the science config reaches the API runtime

## Overview

Two configuration systems exist in `src/colors_of_meaning/shared/` and never meet:

- `shared/synesthetic_config.py` — the SCIENCE/experiment config. `SynestheticConfig`
  composes nested dataclasses (`ProjectorConfig`, `CodebookConfig`, `TrainingConfig`,
  `DistanceConfig`, `DatasetConfig`, `StructuredMapperConfig`, `SupervisedMapperConfig`)
  and is loaded from YAML via `SynestheticConfig.from_yaml`. It is consumed only by the
  CLI: `interface/cli/train.py`, `eval.py`, `encode.py`, `visualize.py`, `compare.py`,
  each calling `SynestheticConfig.from_yaml(args.config)` against files in `configs/`
  (`base.yaml`, `structured.yaml`, `supervised.yaml`).
- `shared/configuration.py` — the API RUNTIME config. `ApplicationSettings`
  (Pydantic `BaseSettings`) carries `admin`, `password`, `reload`, `host`, reads `.env`
  with env prefix `APP_`, then overlays `resources/application.properties` via
  `get_resource_path`. `ApplicationSettingProvider` / `get_application_setting_provider`
  expose it. It is consumed only by `interface/api/main.py` (uvicorn `host`/`reload`)
  and `infrastructure/security/basic_authentication.py` (credentials).

The consequence (the ROADMAP P2-4 finding) is that the experiment a model was trained
under never reaches the running API. `interface/api/main.py`'s `get_container()` wires
only the coconut repositories, the authenticator, and the health checker; it never
imports `synesthetic_config`, never loads a codebook, model, or encoded corpus, and
never registers any of them in the Lagom `Container()`. The one route that genuinely
needs experiment artifacts — `create_query_controller` in
`interface/api/controller/query_controller.py`, which serves `/query/palette` over a
codebook, a distance calculator, and an encoded corpus — is not mounted in `main.py`
(only the coconut and health routers are included). The API therefore has no path by
which a trained `SynestheticConfig` could influence runtime behaviour.

**Recommendation (chosen):** Unify into one coherent settings surface by composing the
science config into the runtime container. Add a single runtime-config component in
`shared/` that resolves an experiment-config path plus the artifact paths
(model, codebook, encoded corpus) the API needs, loads them, and exposes a typed
runtime context. Register that context and the loaded `SynestheticConfig` in the Lagom
`Container()` in `interface/api/main.py`, so the API runtime can load a trained
experiment config the same way the CLI does. Both config modules remain in `shared/`
(importable by every layer); the unification is a composition over them, not a merge
that collapses one into the other. The CLI YAML contract (`SynestheticConfig.from_yaml`
over `configs/*.yaml`) is preserved byte-for-byte — the CLI keeps loading exactly as it
does today.

The alternative — keeping two systems and documenting the boundary with an explicit
bridge — is recorded in Open Questions. It is the smaller change but leaves the API
unable to serve a trained experiment, so it is not the recommendation.

This step is intentionally adjacent to three other P2 items and must not duplicate or
contradict them:

- `011-p2-1-mount-query-retire-coconut` decides whether `create_query_controller` is
  mounted (or deleted) and whether the coconut CRUD is retired. P2-4 supplies the
  config/artifact wiring that a mounted query controller would consume; it does not
  itself decide to mount the route.
- `012-p2-2-hash-credentials` owns the credential fields inside the runtime config
  (`admin`/`password` and `resources/application.properties`). P2-4 must not change the
  credential semantics; it only composes the science config alongside the existing
  runtime settings.
- `013-p2-3-real-health-checks` makes readiness verify codebook/model presence. P2-4's
  resolved artifact paths are the natural single source of truth those checks read, so
  the two must agree on where artifact paths live.

## User Stories

- As an API operator, I want to point the running service at a trained experiment
  (its config plus model/codebook/corpus) so that the API serves results consistent
  with how the model was trained, instead of having no access to the science config.
- As a researcher, I want the API and the CLI to read the same `SynestheticConfig`
  schema so that an experiment I train on the CLI is reproducible behind the API
  without re-specifying projector, codebook, and distance settings.
- As a maintainer, I want one documented settings surface that clearly separates
  runtime concerns (host, reload, credentials) from experiment concerns (projector,
  codebook, distance) so that I can change one without disturbing the other.
- As a reviewer, I want the existing CLI YAML loading to keep working unchanged so that
  this architectural change is back-compatible and low-risk for every CLI command.
- As an operator, I want a clear, fail-fast error when the configured experiment config
  or its artifacts are missing so that misconfiguration surfaces at startup rather than
  on the first request.

## Acceptance Criteria

- [ ] Given a configured experiment-config path, when the API container is built, then
  a `SynestheticConfig` is loaded and registered in the Lagom `Container()` and is
  resolvable by type.
- [ ] Given the runtime config, when the API container is built, then the existing
  runtime settings (`host`, `reload`, and the credential fields owned by
  `012-p2-2-hash-credentials`) continue to resolve exactly as they do today.
- [ ] Given a CLI command (`train`, `eval`, `encode`, `visualize`, `compare`), when it
  loads its `--config` YAML, then `SynestheticConfig.from_yaml` behaves byte-for-byte
  as before and no CLI flag changes meaning (full back-compat).
- [ ] Given the existing `configs/base.yaml`, `configs/structured.yaml`, and
  `configs/supervised.yaml`, when each is parsed after this change, then it parses into
  the same `SynestheticConfig` as before with no schema migration required.
- [ ] Given the runtime config resolves artifact paths (model, codebook, encoded
  corpus), when those paths are read, then they are exposed through a single typed
  accessor that `013-p2-3-real-health-checks` and a mounted query controller can both
  consume (one source of truth for artifact locations).
- [ ] Given an experiment-config path that does not exist or is unreadable, when the
  API container is built, then startup fails fast with a clear error naming the missing
  path, rather than registering a partially built context.
- [ ] Given the runtime context, when its experiment config is overridden for tests
  (in-memory, no disk YAML), then the container can be built without reading
  `configs/` from disk (test isolation; no network or fixed-path I/O in unit tests).
- [ ] Given `interface/api/main.py`, when the module is imported, then the Lagom
  `Container()` — and therefore `get_basic_authenticator()` and the new experiment-config
  resolution — is constructed **once**, not twice, so any fail-closed credential warning
  (owned by `012-p2-2-hash-credentials`) and config-resolution log lines are emitted a
  single time; `create_app()` must still build a container whose artifact loaders
  (`_load_query_codebook`, `_load_corpus`) remain patchable by tests, and `app` must
  remain importable as a module attribute for `uvicorn`.
- [ ] Given this change, when `tox` is run, then all eight gates pass and coverage is
  100%, with one-assertion tests named `test_should_..._when_...` and no comments.

## Hexagonal Layer Impact

### Domain Layer (`src/colors_of_meaning/domain/`)

No changes. `SynestheticConfig` is a configuration/transport structure, not a domain
entity, and stays in `shared/`. No new domain model, port, or service is introduced;
the domain must remain free of configuration-loading concerns.

### Application Layer (`src/colors_of_meaning/application/`)

No changes expected. Use cases (`QueryByPaletteUseCase`, `EncodeDocumentUseCase`,
`EvaluateUseCase`, `TrainColorMappingUseCase`) already receive their collaborators by
constructor injection and stay agnostic of where config comes from. If a later step
needs a use case to read experiment parameters, those are passed in as plain values by
the interface wiring — the application layer must not import either config module's
loading machinery. (If no such need arises here: No changes.)

### Infrastructure Layer (`src/colors_of_meaning/infrastructure/`)

No new adapters required by this step. The artifact loaders the runtime context calls
already exist: `FileColorCodebookRepository` (codebook),
`PyTorchColorMapper.load_weights` (model), and the pickled encoded corpus consumed by
`interface/cli/query.py`. The runtime context composes these existing infrastructure
components; it does not add new ones. `infrastructure/security/basic_authentication.py`
continues to read credentials from the runtime settings unchanged (owned by
`012-p2-2-hash-credentials`). If `013-p2-3-real-health-checks` needs artifact paths,
it reads them from the shared runtime accessor rather than re-deriving them.

### Interface Layer (`src/colors_of_meaning/interface/`)

`interface/api/main.py` is the primary change site. `get_container()` gains, alongside
its current registrations, the loading and registration of the runtime context and the
`SynestheticConfig` it carries, so the API runtime can resolve a trained experiment
config. The CLI entry points (`train.py`, `eval.py`, `encode.py`, `visualize.py`,
`compare.py`) are unchanged — they keep calling `SynestheticConfig.from_yaml(args.config)`
directly. Mounting `create_query_controller` against the newly available context is the
responsibility of `011-p2-1-mount-query-retire-coconut`, not this step; P2-4 only makes
the context resolvable.

Because P2-4 makes `get_container()` materially heavier — it now loads a codebook, a
model, and an encoded corpus — the existing **eager, duplicated** container construction
in `main.py` must be addressed as part of this step. The module builds the container at
import twice: once via the module-level `global_container = get_container()` and again
inside `create_app()` (invoked by the module-level `app = create_app()`). Today that only
repeats cheap wiring; after P2-4 it would load large artifacts twice and — as the
`012-p2-2-hash-credentials` adversarial review found — emit that feature's fail-closed
credential warning twice at import time, before any request is served. Collapse this to a
single construction and emit startup / fail-closed log lines once. The redundant
module-level `global_container = get_container()` is the eager build to remove (expose it
lazily via `get_global_container()`); `create_app()` keeps building its own container so
the query-API tests can still patch `_load_query_codebook` / `_load_corpus` around it, and
`app = create_app()` must remain at module scope so `uvicorn` can import
`colors_of_meaning.interface.api.main:app`.

### Shared Layer

`shared/configuration.py` and `shared/synesthetic_config.py` both remain here and both
remain importable by all layers. The unification is realised as a thin runtime-context
component that composes the two: it resolves the experiment-config path and artifact
paths from the runtime settings (or explicit overrides), calls
`SynestheticConfig.from_yaml`, and exposes a typed runtime context plus a single
artifact-path accessor. Prefer adding this to the existing `shared/configuration.py`
over creating a new module, unless cohesion clearly argues for a dedicated file. The
two existing schemas are not merged into one class; the science schema stays nested
dataclasses with YAML round-tripping, and the runtime schema stays Pydantic
`BaseSettings`.

## API Contracts

No HTTP request/response schema changes in this step. No endpoint is added, removed, or
reshaped here; the `/query/palette` contract in
`interface/api/data_transfer_object/palette_query_dto.py`
(`PaletteQueryRequestDTO` / `PaletteQueryResponseDTO`) is unchanged and remains the
concern of `011-p2-1-mount-query-retire-coconut`. The only API-facing effect is
internal: the container can now resolve a `SynestheticConfig` and a runtime context. Any
endpoint that later returns config-derived data must still return a Pydantic DTO, never
a plain dict, per house rules.

## CLI Impact

No new CLI commands and no changed flags. This is the load-bearing back-compat
guarantee: every CLI command keeps loading its YAML through
`SynestheticConfig.from_yaml(args.config)` exactly as today —
`interface/cli/train.py` (`TrainArgs.config`), `eval.py` (`EvalArgs.config`),
`encode.py` (`EncodeArgs.config`), `visualize.py`, and `compare.py`. The default
`configs/base.yaml` path and the YAML schema are untouched, so existing scripts and
documented commands continue to work without modification.

## Dependency Injection

The API uses a Lagom `Container()` built in `interface/api/main.py::get_container()`.
This step registers the runtime context and the loaded `SynestheticConfig` in that
container by type, alongside the existing `CoconutQueryRepository`,
`CoconutCommandRepository`, use cases, `BasicAuthenticator`, `SecurityDependency`, and
`HealthChecker` registrations. Resolution stays type-based. For tests, the
experiment-config source is injected as an in-memory `SynestheticConfig` (constructed
directly, no disk YAML), so the container builds without reading `configs/` — consistent
with the existing pattern where the container is assembled from explicitly provided
collaborators. The CLI path remains DI-free (factory functions plus constructor
injection) and is not changed.

## Observability

Log experiment-config resolution at container build time with the existing structured
logger and a `correlation-id`: which experiment-config path was loaded and which
artifact paths were resolved (model, codebook, encoded corpus), at INFO; a failed load
at ERROR with the offending path. Do not log credential values from the runtime
settings. No new metrics or tracing spans are required for this step; if added, follow
the existing observability conventions.

## Open Questions

- **Alternative design (documented two-system boundary).** If unification is rejected,
  the fallback is to keep both systems and add an explicit, documented bridge:
  `shared/configuration.py` gains an optional `experiment_config_path` (and artifact
  paths), and a small adapter loads `SynestheticConfig` on demand for the API while the
  CLI is untouched. This is less code but leaves runtime and science config only loosely
  joined. Which approach is sanctioned?
- **Where do artifact paths live?** Should `model_path`, `codebook_name`, and the
  encoded-corpus path be fields on the runtime settings (env/properties driven), be
  read from the loaded `SynestheticConfig`, or be a separate runtime section? This must
  be settled jointly with `013-p2-3-real-health-checks`, which needs the same paths,
  and `011-p2-1-mount-query-retire-coconut`, which needs the encoded corpus.
- **How is the experiment-config path supplied to the API?** Via the `APP_`-prefixed
  env/`application.properties` mechanism already in `ApplicationSettings`, via a CLI
  argument to the API entry point, or both? It must integrate with the existing
  properties+env precedence in `_apply_property`.
- **Strictness when artifacts are absent.** Should the API refuse to start when the
  experiment config is set but a model/codebook/corpus is missing, or start in a
  degraded mode where only config-independent routes (e.g. health, coconut if retained)
  are available? This interacts with the readiness semantics of
  `013-p2-3-real-health-checks`.
- **Multiple experiments at once.** Is a single active experiment sufficient for the
  API runtime, or must it host several (`base`, `structured`, `supervised`)
  concurrently? Multi-experiment hosting would change the runtime context from a single
  value to a keyed registry.
- **Ordering against `011-p2-1`.** Should P2-4 land before `011-p2-1-mount-query-retire-coconut`
  (so the controller has a context to consume when mounted), or together? The two are
  tightly coupled and the merge order should be explicit.
- **Single-vs-eager container construction.** Surfaced by the `012-p2-2-hash-credentials`
  adversarial review: `main.py` builds the container twice at import (module-level
  `global_container` plus the build inside `create_app()`), so startup / fail-closed log
  lines fire twice and, after P2-4, artifacts would load twice. Should the redundant eager
  `global_container = get_container()` simply be removed in favour of a lazy
  `get_global_container()` (smallest change; keeps `create_app()` rebuildable for the
  query-API tests that patch `_load_query_codebook` / `_load_corpus`), or should
  construction move out of import scope entirely into `create_app()` / `run()` behind an
  app factory? The binding constraint is that `uvicorn` imports
  `colors_of_meaning.interface.api.main:app`, so some `app` object must stay
  import-resolvable. This is pre-existing wiring debt (not introduced by P2-4) that P2-4
  is the natural owner of because it is the step that makes the duplicated build expensive.
