from typing import List, Tuple
from unittest.mock import Mock

import numpy as np
import pytest

from colors_of_meaning.application.use_case.train_color_mapping_use_case import TrainColorMappingUseCase


def _build_use_case(scores: List[float]) -> Tuple[TrainColorMappingUseCase, Mock, Mock]:
    mock_color_mapper = Mock()
    mock_color_mapper.epoch_checkpoints.return_value = [f"ckpt{index}" for index in range(len(scores))]
    mock_color_mapper.embed_batch_to_lab.return_value = [Mock(), Mock()]
    mock_evaluator = Mock()
    mock_evaluator.evaluate.side_effect = list(scores)
    mock_codebook_repository = Mock()
    use_case = TrainColorMappingUseCase(mock_color_mapper, mock_evaluator, mock_codebook_repository)
    return use_case, mock_color_mapper, mock_codebook_repository


def _build_use_case_with_factory(scores: List[float]) -> Tuple[TrainColorMappingUseCase, Mock, Mock]:
    mock_color_mapper = Mock()
    mock_color_mapper.epoch_checkpoints.return_value = [f"ckpt{index}" for index in range(len(scores))]
    mock_color_mapper.embed_batch_to_lab.return_value = [Mock(), Mock()]
    mock_evaluator = Mock()
    mock_evaluator.evaluate.side_effect = list(scores)
    mock_codebook_repository = Mock()
    mock_factory = Mock()
    use_case = TrainColorMappingUseCase(mock_color_mapper, mock_evaluator, mock_codebook_repository, mock_factory)
    return use_case, mock_factory, mock_codebook_repository


def _execute_learned(use_case: TrainColorMappingUseCase) -> float:
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return use_case.execute(
        embeddings=embeddings,
        evaluation_embeddings=embeddings,
        epochs=3,
        learning_rate=0.001,
        bins_per_dimension=4,
        model_name="model.pth",
        codebook_name="codebook",
        codebook_mode="learned",
        num_bins=16,
        seed=7,
    )


def _execute(use_case: TrainColorMappingUseCase) -> float:
    embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return use_case.execute(
        embeddings=embeddings,
        evaluation_embeddings=embeddings,
        epochs=3,
        learning_rate=0.001,
        bins_per_dimension=4,
        model_name="model.pth",
        codebook_name="codebook",
    )


class TestTrainColorMappingUseCase:
    def test_should_train_mapper_with_supplied_hyperparameters(self) -> None:
        use_case, mock_color_mapper, _ = _build_use_case([-0.2, -0.9, -0.5])
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        use_case.execute(
            embeddings=embeddings,
            evaluation_embeddings=embeddings,
            epochs=3,
            learning_rate=0.001,
            bins_per_dimension=4,
            model_name="model.pth",
            codebook_name="codebook",
        )

        mock_color_mapper.train.assert_called_once_with(embeddings=embeddings, epochs=3, learning_rate=0.001)

    def test_should_restore_best_scoring_checkpoint_before_saving(self) -> None:
        use_case, mock_color_mapper, _ = _build_use_case([-0.2, -0.9, -0.5])

        _execute(use_case)

        assert mock_color_mapper.restore_checkpoint.call_args_list[-1].args[0] == "ckpt1"

    def test_should_save_best_checkpoint_weights(self) -> None:
        use_case, mock_color_mapper, _ = _build_use_case([-0.2, -0.9, -0.5])

        _execute(use_case)

        mock_color_mapper.save_weights.assert_called_once_with("model.pth")

    def test_should_persist_uniform_codebook(self) -> None:
        use_case, _, mock_codebook_repository = _build_use_case([-0.2, -0.9, -0.5])

        _execute(use_case)

        assert mock_codebook_repository.save.call_args[0][0].num_bins == 64

    def test_should_delegate_to_factory_when_mode_is_learned(self) -> None:
        use_case, mock_factory, _ = _build_use_case_with_factory([-0.2, -0.9, -0.5])

        _execute_learned(use_case)

        mock_factory.build.assert_called_once()

    def test_should_forward_bins_seed_and_embeddings_to_factory_when_mode_is_learned(self) -> None:
        use_case, mock_factory, _ = _build_use_case_with_factory([-0.2, -0.9, -0.5])

        _execute_learned(use_case)

        build_kwargs = mock_factory.build.call_args[1]
        assert build_kwargs["num_bins"] == 16
        assert build_kwargs["seed"] == 7
        assert np.array_equal(build_kwargs["embeddings"], np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    def test_should_save_factory_codebook_when_mode_is_learned(self) -> None:
        use_case, mock_factory, mock_codebook_repository = _build_use_case_with_factory([-0.2, -0.9, -0.5])

        _execute_learned(use_case)

        assert mock_codebook_repository.save.call_args[0][0] is mock_factory.build.return_value

    def test_should_raise_when_learned_mode_has_no_factory(self) -> None:
        use_case, _, _ = _build_use_case([-0.2, -0.9, -0.5])
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        with pytest.raises(ValueError, match="codebook factory is required"):
            use_case.execute(
                embeddings=embeddings,
                evaluation_embeddings=embeddings,
                epochs=3,
                learning_rate=0.001,
                bins_per_dimension=4,
                model_name="model.pth",
                codebook_name="codebook",
                codebook_mode="learned",
            )

    def test_should_return_best_correlation(self) -> None:
        use_case, _, _ = _build_use_case([-0.2, -0.9, -0.5])

        best_correlation = _execute(use_case)

        assert best_correlation == -0.9

    def test_should_restore_the_checkpoint_chosen_by_the_injected_selector(self) -> None:
        mock_color_mapper = Mock()
        selector = Mock(return_value=("chosen", 0.91))
        use_case = TrainColorMappingUseCase(mock_color_mapper, Mock(), Mock(), checkpoint_selector=selector)

        _execute(use_case)

        assert mock_color_mapper.restore_checkpoint.call_args_list[-1].args[0] == "chosen"

    def test_should_return_the_score_from_the_injected_selector(self) -> None:
        selector = Mock(return_value=("chosen", 0.91))
        use_case = TrainColorMappingUseCase(Mock(), Mock(), Mock(), checkpoint_selector=selector)

        assert _execute(use_case) == 0.91
