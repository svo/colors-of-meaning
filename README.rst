.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

====================
Colors of Meaning
====================

Clean architecture meets DDD in a Python-based backend framework with clearly defined boundaries for domain, application, infrastructure, and interface layers.

Architecture
============

This project adopts **Hexagonal Architecture** and **Domain-Driven Design (DDD)** principles, enforcing separation of concerns, testability, and modularity:

- **Domain**: Business logic, entities, and abstract repository interfaces.
- **Application**: Stateless use cases orchestrating the domain.
- **Infrastructure**: Adapters for persistence, observability, messaging, and more.
- **Interface**: RESTful API routes using FastAPI and optional CLI tools.
- **Shared**: Utilities like configuration, formatters, and resilience patterns.

Project Structure Overview
==========================

::

  project/
    application/           # Use cases & orchestration
    domain/                # Business rules & models
    infrastructure/        # Adapters (DB, security, observability)
    interface/             # API (FastAPI) and CLI
    shared/                # Config, utilities, resilience

Design Principles
=================

- **Domain Layer**: Pure logic, independent of frameworks.
- **Application Layer**: Use cases with no external dependencies.
- **Infrastructure Layer**: Implements interfaces; adapters for I/O, caching, messaging, authentication, observability.
- **Interface Layer**: Routes delegate to use cases; Data Transfer Object shape requests/responses.
- **Shared Layer**: Centralized config, formatting, and resilience patterns.

System Qualities
================

- **Performance**: Caching and async messaging.
- **Resilience**: Retry, circuit breakers, fallbacks.
- **Security**: Token-based authentication, auditing, secret management.
- **Observability**: Structured logs, metrics, tracing.
- **Testability**: 100% test coverage, contract and architectural testing.
- **Portability**: Docker, IaC (Terraform, Ansible).

Coding Conventions
==================

- Descriptive naming; self-documenting code (no inline comments).
- Single assertion per test case; test names as clear sentences.
- Strict static analysis enforcement via:

  ===================== ===============================
  Tool                  Purpose
  ===================== ===============================
  flake8                Linting & style
  black                 Formatting
  bandit                Security scanning
  xenon                 Complexity constraints (grade A)
  mypy                  Type checking
  safety / dep-check    Dependency scanning
  semgrep               Static analysis
  ===================== ===============================

Example `xenon` enforcement:

::

  xenon --max-absolute A --max-modules A --max-average A

.. _pyscaffold-notes:

Note
====

This project was set up using PyScaffold 4.6. See https://pyscaffold.org/ for details.
