from unittest.mock import Mock

import numpy as np

from colors_of_meaning.application.use_case.encode_document_use_case import EncodeDocumentUseCase
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.infrastructure.evaluation.validation_accuracy_checkpoint_selector import (
    ValidationAccuracyCheckpointSelector,
)
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)


def _mode_switching_mapper() -> Mock:
    state = {"mode": "collapsing"}
    mapper = Mock()
    mapper.epoch_checkpoints.return_value = ["collapsing", "separating"]
    mapper.restore_checkpoint.side_effect = lambda checkpoint: state.update(mode=checkpoint)

    def embed_batch_to_lab(embeddings: np.ndarray) -> list:
        if state["mode"] == "separating":
            return [LabColor.from_unclamped(50.0, 120.0 if row[0] > 0.5 else -120.0, 0.0) for row in embeddings]
        return [LabColor.from_unclamped(50.0, 0.0, 0.0) for row in embeddings]

    mapper.embed_batch_to_lab.side_effect = embed_batch_to_lab
    return mapper


def _selector(mapper: Mock) -> ValidationAccuracyCheckpointSelector:
    encode_use_case = EncodeDocumentUseCase(QuantizedColorMapper(mapper, ColorCodebook.create_uniform_grid(2)))
    return ValidationAccuracyCheckpointSelector(
        encode_use_case=encode_use_case,
        distance_calculator=JensenShannonDistanceCalculator(smoothing_epsilon=1e-8),
        train_embeddings=np.array([[0.0], [0.0], [1.0], [1.0]], dtype=np.float32),
        train_labels=np.array([0, 0, 1, 1]),
        validation_embeddings=np.array([[0.0], [1.0]], dtype=np.float32),
        validation_labels=np.array([0, 1]),
        k=1,
    )


class TestValidationAccuracyCheckpointSelector:
    def test_should_select_the_checkpoint_with_higher_validation_accuracy(self) -> None:
        mapper = _mode_switching_mapper()

        selected_checkpoint, _accuracy = _selector(mapper)(mapper)

        assert selected_checkpoint == "separating"

    def test_should_report_perfect_validation_accuracy_for_the_separating_checkpoint(self) -> None:
        mapper = _mode_switching_mapper()

        _selected_checkpoint, accuracy = _selector(mapper)(mapper)

        assert accuracy == 1.0
