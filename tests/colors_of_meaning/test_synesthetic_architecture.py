from pytest_archon.rule import archrule


def test_should_maintain_domain_layer_independence() -> None:
    (
        archrule(
            "Domain Layer Independence",
            comment="Domain layer should not depend on other layers (Clean Architecture)",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "colors_of_meaning.infrastructure.*",
            "colors_of_meaning.interface.*",
            "colors_of_meaning.application.*",
        )
        .check("colors_of_meaning")
    )


def test_should_maintain_application_interface_independence() -> None:
    (
        archrule(
            "Application Interface",
            comment="Application layer should not depend on interface layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.interface.*")
        .check("colors_of_meaning")
    )


def test_should_maintain_application_infrastructure_independence() -> None:
    (
        archrule(
            "Application Infrastructure",
            comment="Application layer should not depend on infrastructure layer",
        )
        .match("colors_of_meaning.application.*")
        .should_not_import("colors_of_meaning.infrastructure.*")
        .check("colors_of_meaning")
    )


def test_should_use_application_use_case_in_cli() -> None:
    (
        archrule(
            "CLI Use Case",
            comment="Interface CLI commands should depend on application use cases",
        )
        .match(
            "colors_of_meaning.interface.cli.train",
            "colors_of_meaning.interface.cli.encode",
            "colors_of_meaning.interface.cli.compare",
            "colors_of_meaning.interface.cli.compress",
            "colors_of_meaning.interface.cli.query",
        )
        .should_import("colors_of_meaning.application.use_case.*")
        .check("colors_of_meaning")
    )


def test_should_keep_dataset_infrastructure_independent_of_upper_layers() -> None:
    (
        archrule(
            "Dataset Infrastructure Independence",
            comment="Infrastructure dataset adapters and the seeded sampler must not import application or interface",
        )
        .match("colors_of_meaning.infrastructure.dataset.*")
        .should_not_import(
            "colors_of_meaning.application.*",
            "colors_of_meaning.interface.*",
        )
        .check("colors_of_meaning")
    )


def test_should_keep_structure_metric_numerical_dependencies_out_of_domain() -> None:
    (
        archrule(
            "Structure Metric Dependency Isolation",
            comment="The scipy-backed structure-preservation metric must stay out of the domain layer",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "scipy",
            "scipy.*",
            "torch",
            "torch.*",
        )
        .check("colors_of_meaning")
    )


def test_should_not_import_sklearn_in_domain_layer() -> None:
    (
        archrule(
            "Domain k-means Isolation",
            comment="The learned-codebook k-means dependency must stay out of the domain layer",
        )
        .match("colors_of_meaning.domain.*")
        .should_not_import(
            "sklearn",
            "sklearn.*",
        )
        .check("colors_of_meaning")
    )


def test_should_confine_structure_preservation_evaluator_scipy_to_infrastructure() -> None:
    (
        archrule(
            "Structure Preservation Evaluator Infrastructure",
            comment="The Spearman evaluator implementation lives in infrastructure and may depend on scipy",
        )
        .match("colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator")
        .should_import("scipy.stats")
        .check("colors_of_meaning")
    )


def test_should_maintain_shared_module_independence() -> None:
    (
        archrule(
            "Shared Module Dependencies",
            comment="Shared module should not depend on application, infrastructure or interface",
        )
        .match("colors_of_meaning.shared.*")
        .should_not_import(
            "colors_of_meaning.application.*",
            "colors_of_meaning.infrastructure.*",
            "colors_of_meaning.interface.*",
        )
        .check("colors_of_meaning")
    )


def test_should_not_have_circular_dependencies() -> None:
    (
        archrule(
            "No Circular Dependencies",
            comment="No modules should have circular dependencies",
        )
        .match("colors_of_meaning.*")
        .should(
            lambda module, direct_imports, all_imports: module not in direct_imports
            and module not in all_imports.get(module, set()),
            "no_circular_dependencies",
        )
        .check("colors_of_meaning")
    )
