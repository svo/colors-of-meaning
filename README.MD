# Python Sprint Zero

[![Build Builder](https://github.com/svo/python-sprint-zero/actions/workflows/builder.yml/badge.svg)](https://github.com/svo/python-sprint-zero/actions/workflows/builder.yml)
[![Build Development](https://github.com/svo/python-sprint-zero/actions/workflows/development.yml/badge.svg)](https://github.com/svo/python-sprint-zero/actions/workflows/development.yml)
[![Build Service](https://github.com/svo/python-sprint-zero/actions/workflows/service.yml/badge.svg)](https://github.com/svo/python-sprint-zero/actions/workflows/service.yml)

## Development

### Tested Configuration

* `vagrant`: 2.4.3
* `ansible`: 2.18.3
* `colima`: 0.8.1
* `docker`: 28.0.2

## Instructions

```
colima start --cpu 2 --memory 2 --vm-type=vz --mount-type=virtiofs --dns 8.8.8.8
vagrant up
vagrant provision
vagrant ssh
```

### Test

#### Shell Script

```
shellcheck -x ./*.sh bin/*
```

#### Ansible

```
ansible-lint -p infrastructure/ansible/playbook-*.yml
```

#### Python

```
tox
```

##### Watch

```
tox -e watch
```

### Code Formatter

```
tox -e format
```

### Run

```
./bin/run-local -c
```

## Architecture

The architectural structure and design rules for this project uses a combination of Hexagonal Architecture and Domain-Driven Design (`DDD`) principles. The architecture is organized into clearly separated domain, application, infrastructure, and interface layers, following clean architecture principles. The domain layer encapsulates entities and business rules, the application layer manages orchestration through use cases, infrastructure contains data persistence implementations with repositories, and the interface layer provides `RESTful` `API` routes. The structure emphasizes high test coverage, maintainability, and security by design.

---

### Project Structure Overview

```
project/

 application/                       # Use cases
    use_case/
        coconut_use_case.py
    service/
        example_service.py

 domain/                            # Business logic
    authentication/
        authentitator.py
    health/
        health_status.py
    model/
        coconut.py
    repository/
        coconut_repository.py
    service/
        example_data_import_service.py

 infrastructure/                    # Adapters, drivers
    observability/
        logger.py
        metrics.py
        tracing.py
    persistence/
        in_memory_coconut_command_repository.py
        in_memory_coconut_query_repository.py
    cache/
        cache_provider.py
    message/
        message_producer.py
        message_consumer.py
    importer/
        example_importer.py
        base_importer.py
    security/
        basic_authentication.py
    system/
        health_checker.py

 interface/                         # APIs and CLI
    api/
        main.py
        controller/
            coconut_controller.py
            health_controller.py
        data_transfer_object/
            coconut_data_transfer_object.py
            health_status_data_transfer_object.py

 shared/                            # Cross-cutting concerns
    configuration.py
    resilience/
        retry.py
        circuit_breaker.py
    formatter/
        example_formatter.py
```

---

### Design Principles

#### Domain

- `model/` contain pure domain entities (e.g., `DataRecord`)
- `repository/` define interfaces for persistence (abstract base classes)
- `service/` contain stateless domain logic (business rules)

#### Application

- Use cases (`_use_case.py`) orchestrate logic/delegate
- Should not directly depend on e.g. `FastAPI`, database, or file system

#### Infrastructure

- Implements interfaces from `domain.repository`
- `importer/` handles imports
- `security/` provides token-based authentication
- `system/` contains diagnostics (e.g., disk, database checks)
- `observability/` provides logging, tracing decorators

#### Interface

- `api/controller/` expose `FastAPI` routes and depend on use cases
- `api/data_transfer_object/` define `Pydantic` model for request/response shaping
- `cli/` optionally runs scripts or command-line tools

#### Shared

- `formatter/` contains reusable utility classes for formatting (e.g., date/time conversion)
- `configuration.py` loads settings

---

### Dependency Injection

This project uses the dependency injection pattern to enhance modularity, testability, and maintainability, with [Lagom](https://github.com/meadsteve/lagom) as the dependency injection framework.

#### Key Principles

- **Inversion of Control**: Dependencies are provided to components rather than created internally.
- **Interface-based Programming**: Components depend on abstractions (interfaces) rather than concrete implementations.
- **Single Responsibility**: Each component focuses on its core functionality, with dependencies injected as needed.
- **Testability**: Allows easy replacement of dependencies with test doubles (mocks, stubs) in unit tests.

#### Using `Lagom` for Dependency Injection

`Lagom` provides a lightweight, Pythonic approach to dependency injection with features that align well with our architecture:

- **Type-based Resolution**: Dependencies are resolved based on type annotations.
- **Singletons and Transients**: Support for both singleton (shared) and transient (new instance per request) dependencies.
- **Contextual Containers**: Create specialized containers for different application contexts (e.g., testing vs. production).

---

### Enhancing System Quality

#### Performance and Scalability

- Implement caching strategies (`Redis`) for frequently accessed data.
- Use message queues (`Pub/Sub`) for asynchronous tasks.

#### Reliability and Fault Tolerance

- Explicitly define retry and circuit breaker strategies.
- Clearly document error handling and recovery procedures.

#### Maintainability and Modularity

- Clearly define module boundaries and use explicit interfaces (`ABC`).

#### Observability and Monitoring

- Structured logging with `correlation-id`.
- Metrics collection and distributed tracing.

#### Security

- Auditing of key domain events.
- Secure management of secrets (`Vault`).

#### Availability

- Explicit fall-back or degraded-service strategies.
- Robust health-check mechanisms.

#### Testability

- Include integration and end-to-end tests for core functionality.
- Contract testing for integrations.

#### Portability

- Containerization strategy (`Docker`).
- Infrastructure as code (`Terraform`, `Ansible`, `Packer`).

---

### Coding Conventions

#### General

- No comments: code should be self-documenting and expressive
- Descriptive naming: variables, functions, and classes should clearly communicate their intent
- Include `__init__.py` files in each Python package to clearly define module boundaries and support tools like `linters`, test runners, and code navigation.

#### Testing

- 100% test coverage is expected
- Tests should be meaningful and not just aim to increase coverage
- Mocks/stubs should be used where necessary to isolate behaviour
- Tests should include only one assertion per test case
- Test names should be phrased as a sentence clearly reflecting the behaviour being asserted
  - **Example:** `test_should_return_404_when_resource_is_not_found()`
- Consumer Driven Contract Testing (`CDCT`) consumer tests are required for any internal service called, and producer tests for `API` routes provided by this project.
- Architectural Unit Testing should be used to assert that the code aligns with the principles defined in this document.

#### Static Analysis

Code must pass the following tools before merging:

| Tool                | Purpose                                            |
|---------------------|----------------------------------------------------|
| `flake8`            | `Linting` and style checking                       |
| `black`             | Code formatting                                    |
| `bandit`            | Security scanning for common issues                |
| `xenon`             | Maintainability and complexity limits              |
| `mypy`              | Static type checking                               |
| `semgrep`           | Static analysis for code patterns and security     |
| `pip-audit`         | Dependency vulnerability scanning                  |

---

By following these conventions, the code remains:

- Clean and easy to reason about
- Secure by default
- Readable by both humans and tools
- Easy to `refactor` and scale with confidence
- Loosely coupled through dependency injection
