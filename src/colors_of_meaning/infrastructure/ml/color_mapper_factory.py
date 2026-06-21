from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)
from colors_of_meaning.shared.synesthetic_config import SynestheticConfig


def create_color_mapper(mapper_type: str, config: SynestheticConfig) -> ColorMapper:
    if mapper_type == "structured":
        return _create_structured_mapper(config)
    if mapper_type == "supervised":
        return _create_supervised_mapper(config)
    if mapper_type == "unconstrained":
        return _create_unconstrained_mapper(config)
    raise ValueError(f"Unknown mapper type: {mapper_type}. Supported: unconstrained, structured, supervised")


def _create_unconstrained_mapper(config: SynestheticConfig) -> ColorMapper:
    return PyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
    )


def _create_structured_mapper(config: SynestheticConfig) -> ColorMapper:
    structured_config = config.structured_mapper
    if structured_config is None:
        raise ValueError("structured_mapper config is required for structured mapper type")
    return StructuredPyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
        alpha=structured_config.alpha,
        beta=structured_config.beta,
        gamma=structured_config.gamma,
        num_clusters=structured_config.num_clusters,
        max_chroma=structured_config.max_chroma,
    )


def _create_supervised_mapper(config: SynestheticConfig) -> ColorMapper:
    supervised_config = config.supervised_mapper
    if supervised_config is None:
        raise ValueError("supervised_mapper config is required for supervised mapper type")
    return SupervisedPyTorchColorMapper(
        input_dim=config.projector.embedding_dim,
        hidden_dim_1=config.projector.hidden_dim_1,
        hidden_dim_2=config.projector.hidden_dim_2,
        dropout_rate=config.projector.dropout_rate,
        device=config.training.device,
        num_classes=supervised_config.num_classes,
        classification_weight=supervised_config.classification_weight,
    )
