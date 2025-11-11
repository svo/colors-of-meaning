from pytest_archon.rule import archrule


def test_should_maintain_domain_layer_independence():
    (
        archrule(
            "Domain Layer Independence",
            comment="Domain layer should not depend on other layers (Clean Architecture)",
        )
        .match("python_sprint_zero.domain.*")
        .should_not_import(
            "python_sprint_zero.infrastructure.*",
            "python_sprint_zero.interface.*",
            "python_sprint_zero.application.*",
        )
        .check("python_sprint_zero")
    )


def test_should_maintain_application_interface_independence():
    (
        archrule(
            "Application Interface",
            comment="Application layer should not depend on interface layer",
        )
        .match("python_sprint_zero.application.*")
        .should_not_import("python_sprint_zero.interface.*")
        .check("python_sprint_zero")
    )


def test_should_maintain_application_infrastructure_independence():
    (
        archrule(
            "Application Infrastructure",
            comment="Application layer should not depend on infrastructure layer",
        )
        .match("python_sprint_zero.application.*")
        .should_not_import("python_sprint_zero.infrastructure.*")
        .check("python_sprint_zero")
    )


def test_should_use_application_use_case_in_controller():
    (
        archrule(
            "Controller Use Case",
            comment="Interface controller should depend on application use cases",
        )
        .match("python_sprint_zero.interface.api.controller.*")
        .should_import("python_sprint_zero.application.use_case.*")
        .check("python_sprint_zero")
    )


def test_should_not_use_domain_model_in_data_transfer_object():
    (
        archrule(
            "Data Transfer Object Model",
            comment="Data Transfer Object should not depend directly on domain model",
        )
        .match("python_sprint_zero.interface.api.data_transfer_object.*")
        .should_not_import("python_sprint_zero.domain.model.*")
        .check("python_sprint_zero")
    )


def test_should_follow_security_module_architecture():
    (
        archrule(
            "Security Module",
            comment="Security components should follow architectural boundaries",
        )
        .match("python_sprint_zero.infrastructure.security.*")
        .should_import("python_sprint_zero.domain.authentication.*")
        .check("python_sprint_zero")
    )


def test_should_not_have_circular_dependencies():
    (
        archrule(
            "No Circular Dependencies",
            comment="No modules should have circular dependencies",
        )
        .match("python_sprint_zero.*")
        .should(
            lambda module, direct_imports, all_imports: module not in direct_imports
            and module not in all_imports.get(module, set()),
            "no_circular_dependencies",
        )
        .check("python_sprint_zero")
    )


def test_should_maintain_shared_module_independence():
    (
        archrule(
            "Shared Module Dependencies",
            comment="Shared module should not depend on application, infrastructure or interface",
        )
        .match("python_sprint_zero.shared.*")
        .should_not_import(
            "python_sprint_zero.application.*", "python_sprint_zero.infrastructure.*", "python_sprint_zero.interface.*"
        )
        .check("python_sprint_zero")
    )
