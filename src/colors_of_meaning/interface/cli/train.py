import logging
import uuid
import numpy as np
import numpy.typing as npt
import tyro
from dataclasses import dataclass

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig
from colors_of_meaning.shared.determinism import seed_everything
from colors_of_meaning.infrastructure.embedding.sentence_embedding_adapter import (
    SentenceEmbeddingAdapter,
)
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.evaluation.structure_preservation_evaluator import (
    SpearmanStructurePreservationEvaluator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.application.use_case.train_color_mapping_use_case import (
    TrainColorMappingUseCase,
)
from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import (
    AGNewsDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import (
    IMDBDatasetAdapter,
)
from colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter import (
    NewsgroupsDatasetAdapter,
)
from colors_of_meaning.domain.repository.dataset_repository import DatasetRepository

logger = logging.getLogger(__name__)


@dataclass
class TrainArgs:
    config: str = "configs/base.yaml"
    dataset_path: str = "data/train.txt"
    output_model: str = "artifacts/models/projector.pth"
    output_codebook: str = "codebook_4096"
    mapper_type: str = "unconstrained"
    deterministic: bool = False


def _create_dataset_adapter(dataset_name: str) -> DatasetRepository:
    adapters: dict[str, type[DatasetRepository]] = {
        "ag_news": AGNewsDatasetAdapter,
        "imdb": IMDBDatasetAdapter,
        "20newsgroups": NewsgroupsDatasetAdapter,
    }
    adapter_class = adapters.get(dataset_name)
    if adapter_class is None:
        raise ValueError(f"Unknown dataset: {dataset_name}. Supported: {list(adapters.keys())}")
    return adapter_class()


def _create_color_mapper(args: TrainArgs, config: SynestheticConfig) -> ColorMapper:
    return create_color_mapper(args.mapper_type, config)


def _load_supervised_data(config: SynestheticConfig) -> tuple:
    dataset_adapter = _create_dataset_adapter(config.dataset.name)
    samples = dataset_adapter.get_samples(
        split=config.dataset.train_split,
        max_samples=config.dataset.max_samples,
        seed=config.training.seed,
    )
    texts = [sample.text for sample in samples]
    labels = np.array([sample.label for sample in samples])
    return texts, labels


def _load_texts_from_file(dataset_path: str) -> list:
    with open(dataset_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _configure_supervised_mapper(
    color_mapper: ColorMapper,
    labels: npt.NDArray,
) -> None:
    if isinstance(color_mapper, SupervisedPyTorchColorMapper):
        color_mapper.set_training_labels(labels)


def _apply_determinism(args: TrainArgs, config: SynestheticConfig) -> None:
    seed_everything(config.training.seed, deterministic=args.deterministic)
    logger.info(
        "Applied training determinism settings",
        extra={
            "correlation_id": str(uuid.uuid4()),
            "seed": config.training.seed,
            "deterministic": args.deterministic,
        },
    )


def _select_evaluation_embeddings(
    embeddings: npt.NDArray,
    seed: int,
    max_evaluation_samples: int = 256,
) -> npt.NDArray:
    generator = np.random.default_rng(seed)
    sample_count = min(len(embeddings), max_evaluation_samples)
    selection = generator.permutation(len(embeddings))[:sample_count]
    return np.asarray(embeddings[selection])


def main(args: TrainArgs) -> None:
    config = SynestheticConfig.from_yaml(args.config)
    _apply_determinism(args, config)
    labels = None

    if args.mapper_type == "supervised":
        print(f"Loading labeled dataset: {config.dataset.name}...")
        texts, labels = _load_supervised_data(config)
    else:
        print(f"Loading dataset from {args.dataset_path}...")
        texts = _load_texts_from_file(args.dataset_path)

    print(f"Encoding {len(texts)} texts with sentence embeddings...")
    embedding_adapter = SentenceEmbeddingAdapter()
    embeddings = embedding_adapter.encode_batch(texts, batch_size=config.training.batch_size, show_progress=True)

    color_mapper = _create_color_mapper(args, config)

    if labels is not None:
        _configure_supervised_mapper(color_mapper, labels)

    _execute_training(args, config, color_mapper, embeddings)


def _execute_training(
    args: TrainArgs,
    config: SynestheticConfig,
    color_mapper: ColorMapper,
    embeddings: npt.NDArray,
) -> None:
    print(f"Training {args.mapper_type} color projector for {config.training.epochs} epochs...")
    codebook_repo = FileColorCodebookRepository()
    evaluator = SpearmanStructurePreservationEvaluator(seed=config.training.seed)
    evaluation_embeddings = _select_evaluation_embeddings(embeddings, config.training.seed)
    use_case = TrainColorMappingUseCase(
        color_mapper=color_mapper,
        structure_preservation_evaluator=evaluator,
        codebook_repository=codebook_repo,
    )

    correlation = use_case.execute(
        embeddings=embeddings,
        evaluation_embeddings=evaluation_embeddings,
        epochs=config.training.epochs,
        learning_rate=config.training.learning_rate,
        bins_per_dimension=config.codebook.bins_per_dimension,
        model_name=args.output_model,
        codebook_name=args.output_codebook,
    )

    print(f"Model saved to {args.output_model}")
    print(f"Codebook saved to artifacts/codebooks/{args.output_codebook}.pkl")
    print(f"Structure-preservation correlation: {correlation:.4f}")


if __name__ == "__main__":
    main(tyro.cli(TrainArgs))
