from unittest.mock import Mock
import numpy as np

from colors_of_meaning.application.use_case.train_color_mapping_use_case import TrainColorMappingUseCase


class TestTrainColorMappingUseCase:
    def test_should_train_mapper_and_save_artifacts(self) -> None:
        mock_color_mapper = Mock()
        mock_codebook_repository = Mock()

        use_case = TrainColorMappingUseCase(mock_color_mapper, mock_codebook_repository)
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        use_case.execute(
            embeddings=embeddings,
            epochs=10,
            learning_rate=0.001,
            bins_per_dimension=4,
            model_name="test_model.pth",
            codebook_name="test_codebook",
        )

        mock_color_mapper.train.assert_called_once_with(embeddings=embeddings, epochs=10, learning_rate=0.001)
        mock_color_mapper.save_weights.assert_called_once_with("test_model.pth")
        mock_codebook_repository.save.assert_called_once()

        saved_codebook = mock_codebook_repository.save.call_args[0][0]
        assert saved_codebook.num_bins == 64
