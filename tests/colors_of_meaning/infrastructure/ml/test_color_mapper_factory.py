from pathlib import Path

import pytest
from unittest.mock import Mock

from colors_of_meaning.infrastructure.ml.color_mapper_factory import create_color_mapper
from colors_of_meaning.infrastructure.ml.pytorch_color_mapper import PyTorchColorMapper
from colors_of_meaning.infrastructure.ml.structured_pytorch_color_mapper import (
    StructuredPyTorchColorMapper,
)
from colors_of_meaning.infrastructure.ml.supervised_pytorch_color_mapper import (
    SupervisedPyTorchColorMapper,
)


class TestCreateColorMapper:
    @pytest.fixture
    def config(self) -> Mock:
        mock_config = Mock()
        mock_config.projector.embedding_dim = 10
        mock_config.projector.hidden_dim_1 = 8
        mock_config.projector.hidden_dim_2 = 4
        mock_config.projector.dropout_rate = 0.1
        mock_config.training.device = "cpu"
        mock_config.training.seed = 42
        mock_config.structured_mapper.alpha = 1.0
        mock_config.structured_mapper.beta = 1.0
        mock_config.structured_mapper.gamma = 1.0
        mock_config.structured_mapper.num_clusters = 16
        mock_config.structured_mapper.max_chroma = 128.0
        mock_config.supervised_mapper.classification_weight = 0.1
        mock_config.supervised_mapper.num_classes = 4
        return mock_config

    def test_should_construct_unconstrained_mapper_when_mapper_type_is_unconstrained(self, config: Mock) -> None:
        mapper = create_color_mapper("unconstrained", config)

        assert isinstance(mapper, PyTorchColorMapper)

    def test_should_construct_structured_mapper_when_mapper_type_is_structured(self, config: Mock) -> None:
        mapper = create_color_mapper("structured", config)

        assert isinstance(mapper, StructuredPyTorchColorMapper)

    def test_should_construct_supervised_mapper_when_mapper_type_is_supervised(self, config: Mock) -> None:
        mapper = create_color_mapper("supervised", config)

        assert isinstance(mapper, SupervisedPyTorchColorMapper)

    def test_should_raise_value_error_when_structured_config_is_none(self, config: Mock) -> None:
        config.structured_mapper = None

        with pytest.raises(ValueError):
            create_color_mapper("structured", config)

    def test_should_raise_value_error_when_supervised_config_is_none(self, config: Mock) -> None:
        config.supervised_mapper = None

        with pytest.raises(ValueError):
            create_color_mapper("supervised", config)

    def test_should_raise_value_error_when_mapper_type_is_unknown(self, config: Mock) -> None:
        with pytest.raises(ValueError):
            create_color_mapper("nonexistent", config)

    def test_should_load_checkpoint_without_error_when_mapper_type_is_structured(
        self, config: Mock, tmp_path: Path
    ) -> None:
        checkpoint = StructuredPyTorchColorMapper(
            input_dim=10, hidden_dim_1=8, hidden_dim_2=4, dropout_rate=0.1, device="cpu"
        )
        path = str(tmp_path / "structured.pth")
        checkpoint.save_weights(path)

        mapper = create_color_mapper("structured", config)
        mapper.load_weights(path)

        assert isinstance(mapper, StructuredPyTorchColorMapper)

    def test_should_load_checkpoint_without_error_when_mapper_type_is_supervised(
        self, config: Mock, tmp_path: Path
    ) -> None:
        checkpoint = SupervisedPyTorchColorMapper(
            input_dim=10, hidden_dim_1=8, hidden_dim_2=4, dropout_rate=0.1, device="cpu", num_classes=4
        )
        path = str(tmp_path / "supervised.pth")
        checkpoint.save_weights(path)

        mapper = create_color_mapper("supervised", config)
        mapper.load_weights(path)

        assert isinstance(mapper, SupervisedPyTorchColorMapper)
