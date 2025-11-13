from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

from colors_of_meaning.interface.cli.train import main, TrainArgs


class TestTrainCLI:
    @patch("colors_of_meaning.interface.cli.train.SynestheticConfig")
    @patch("colors_of_meaning.interface.cli.train.SentenceEmbeddingAdapter")
    @patch("colors_of_meaning.interface.cli.train.PyTorchColorMapper")
    @patch("colors_of_meaning.interface.cli.train.FileColorCodebookRepository")
    @patch("colors_of_meaning.interface.cli.train.TrainColorMappingUseCase")
    @patch("builtins.open", new_callable=mock_open, read_data="text1\ntext2\ntext3\n")
    @patch("builtins.print")
    def test_should_execute_training_workflow(
        self,
        mock_print: Mock,
        mock_file: Mock,
        mock_use_case_class: Mock,
        mock_repo_class: Mock,
        mock_mapper_class: Mock,
        mock_adapter_class: Mock,
        mock_config_class: Mock,
        tmp_path: Path,
    ) -> None:
        mock_config = Mock()
        mock_config.training.batch_size = 32
        mock_config.training.epochs = 10
        mock_config.training.learning_rate = 0.001
        mock_config.training.device = "cpu"
        mock_config.codebook.bins_per_dimension = 4
        mock_config.projector.embedding_dim = 384
        mock_config.projector.hidden_dim_1 = 128
        mock_config.projector.hidden_dim_2 = 64
        mock_config.projector.dropout_rate = 0.1
        mock_config_class.from_yaml.return_value = mock_config

        mock_adapter = Mock()
        mock_adapter.encode_batch.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_adapter_class.return_value = mock_adapter

        mock_mapper = Mock()
        mock_mapper_class.return_value = mock_mapper

        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        mock_use_case = Mock()
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        dataset_path = tmp_path / "train.txt"
        output_model = tmp_path / "model.pth"
        output_codebook = tmp_path / "codebook"

        config_path.write_text("dummy")
        dataset_path.write_text("text1\ntext2\ntext3\n")

        args = TrainArgs(
            config=str(config_path),
            dataset_path=str(dataset_path),
            output_model=str(output_model),
            output_codebook=str(output_codebook),
        )

        main(args)

        mock_config_class.from_yaml.assert_called_once_with(str(config_path))
        mock_adapter.encode_batch.assert_called_once()
        mock_use_case.execute.assert_called_once()
