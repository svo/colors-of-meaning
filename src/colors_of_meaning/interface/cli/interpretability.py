import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, cast

import tyro

from colors_of_meaning.application.use_case.evaluate_interpretability_use_case import (
    EvaluateInterpretabilityUseCase,
)
from colors_of_meaning.domain.model.interpretability_report import (
    INTERPRETABILITY_AXES,
    InterpretabilityReport,
    InterpretabilityThresholds,
)
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import (
    IMDBDatasetAdapter,
)
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.infrastructure.evaluation.sklearn_interpretability_evaluator import (
    SklearnInterpretabilityEvaluator,
)
from colors_of_meaning.infrastructure.ml.brysbaert_concreteness_lexicon import (
    BrysbaertConcretenessLexicon,
)
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.shared.synesthetic_config import (
    StructuredMapperConfig,
    SynestheticConfig,
)

logger = logging.getLogger(__name__)

AXIS_LABELS: Dict[str, str] = {
    "hue_topic": "hue <-> topic (NMI)",
    "lightness_sentiment": "lightness <-> sentiment (corr)",
    "chroma_concreteness": "chroma <-> concreteness (corr)",
}


@dataclass
class InterpretabilityArgs:
    config: str = "configs/interpretability.yaml"
    dataset: Literal["imdb"] = "imdb"
    structured_model: str = "artifacts/models/structured_projector.pth"
    control: Literal["unconstrained", "noise"] = "noise"
    control_model: str = "artifacts/models/projector.pth"
    max_samples: Optional[int] = 500
    hue_topic_margin: float = 0.02
    lightness_sentiment_margin: float = 0.05
    chroma_concreteness_margin: float = 0.05
    output_path: str = "reports/interpretability.md"


def _setup_dataset(dataset_name: str) -> DatasetRepository:
    adapters: Dict[str, type[DatasetRepository]] = {"imdb": IMDBDatasetAdapter}
    return adapters[dataset_name]()


def _structured_config(config: SynestheticConfig) -> StructuredMapperConfig:
    return cast(StructuredMapperConfig, config.structured_mapper)


def _build_structured_mapper(args: InterpretabilityArgs, config: SynestheticConfig) -> ColorMapper:
    mapper = create_color_mapper("structured", config)
    mapper.load_weights(args.structured_model)
    return mapper


def _build_control_mapper(args: InterpretabilityArgs, config: SynestheticConfig) -> ColorMapper:
    mapper = create_color_mapper("unconstrained", config)
    if args.control == "unconstrained":
        mapper.load_weights(args.control_model)
    return mapper


def _build_thresholds(args: InterpretabilityArgs) -> InterpretabilityThresholds:
    return InterpretabilityThresholds(
        hue_topic_margin=args.hue_topic_margin,
        lightness_sentiment_margin=args.lightness_sentiment_margin,
        chroma_concreteness_margin=args.chroma_concreteness_margin,
    )


def _build_use_case(args: InterpretabilityArgs, config: SynestheticConfig) -> EvaluateInterpretabilityUseCase:
    structured_config = _structured_config(config)
    return EvaluateInterpretabilityUseCase(
        embedding_adapter=SentenceEmbeddingAdapter(),
        structured_mapper=_build_structured_mapper(args, config),
        control_mapper=_build_control_mapper(args, config),
        interpretability_evaluator=SklearnInterpretabilityEvaluator(num_hue_bins=structured_config.num_clusters),
        concreteness_lexicon=BrysbaertConcretenessLexicon(resource_name=structured_config.concreteness_resource),
        thresholds=_build_thresholds(args),
    )


def _verdict(report: InterpretabilityReport) -> str:
    return "VALIDATED" if report.is_validated else "FALSIFIED"


def _falsified_summary(report: InterpretabilityReport) -> str:
    return ", ".join(report.falsified_axes) if report.falsified_axes else "none"


def _axis_rows(report: InterpretabilityReport) -> List[str]:
    structured = report.structured.axis_scores
    control = report.control.axis_scores
    margins = report.margins
    thresholds = report.thresholds.axis_margins
    rows = [
        "| axis | structured | control | margin | threshold | verdict |",
        "|------|-----------|---------|--------|-----------|---------|",
    ]
    for axis in INTERPRETABILITY_AXES:
        outcome = "falsified" if axis in report.falsified_axes else "pass"
        rows.append(
            f"| {AXIS_LABELS[axis]} | {structured[axis]:.4f} | {control[axis]:.4f} | "
            f"{margins[axis]:+.4f} | {thresholds[axis]:.4f} | {outcome} |"
        )
    return rows


def _provenance_line() -> str:
    import numpy
    import scipy  # type: ignore
    import sklearn  # type: ignore

    return (
        f"Library versions: numpy {numpy.__version__}, scipy {scipy.__version__}, scikit-learn {sklearn.__version__}."
    )


def _reproduce_command(args: InterpretabilityArgs) -> str:
    return (
        f"tox -e interpretability -- --dataset {args.dataset} --structured-model {args.structured_model} "
        f"--control {args.control} --max-samples {args.max_samples} --config {args.config}"
    )


def _report_lines(args: InterpretabilityArgs, report: InterpretabilityReport) -> List[str]:
    return [
        "# Interpretability validation of the structured color mapper",
        "",
        "The structured mapper is *trained toward* hue=topic-cluster, lightness=sentiment, and",
        "chroma=concreteness targets, so the claim is true by construction on the training split.",
        "This report makes it falsifiable: each axis is measured on a held-out split for the",
        "structured mapper and for a negative control (a mapper never trained toward these axes).",
        "Interpretability is asserted only where the structured margin over the control clears the",
        "documented threshold; any axis that fails is reported as a falsification, not a pass.",
        "",
        "Method: each document's Lab color is read from its document embedding (the same granularity",
        "the mapper is trained at); the sentiment signal is the IMDB binary label; concreteness is the",
        "bundled Brysbaert lexicon score (offline). hue<->topic is normalized mutual information of",
        "binned hue angle vs the gold class; the other two axes are rank/point-biserial correlation.",
        "",
        _provenance_line(),
        "",
        f"Overall verdict: **{_verdict(report)}**.",
        f"Falsified axes: {_falsified_summary(report)}.",
        "",
        "## Per-axis scores",
        "",
        *_axis_rows(report),
        "",
        "## Reproduce",
        "",
        "```bash",
        _reproduce_command(args),
        "```",
        "",
    ]


def _write_report(output_path: str, args: InterpretabilityArgs, report: InterpretabilityReport) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(_report_lines(args, report)), encoding="utf-8")


def _print_table(report: InterpretabilityReport) -> None:
    print("\n=== Interpretability Validation ===")
    for line in _axis_rows(report):
        print(line)
    print(f"\nVerdict: {_verdict(report)} (falsified axes: {_falsified_summary(report)})")


def _log_startup(args: InterpretabilityArgs, config: SynestheticConfig) -> None:
    logger.info(
        "Starting interpretability validation",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "dataset": args.dataset,
            "control": args.control,
            "seed": config.training.seed,
        },
    )


def main(args: InterpretabilityArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    _log_startup(args, config)
    use_case = _build_use_case(args, config)
    dataset = _setup_dataset(args.dataset)
    samples = dataset.get_samples(
        split=config.dataset.test_split, max_samples=args.max_samples, seed=config.training.seed
    )
    report = use_case.execute(samples)
    _print_table(report)
    _write_report(args.output_path, args, report)


if __name__ == "__main__":
    main(tyro.cli(InterpretabilityArgs))
