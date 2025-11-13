from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import numpy as np

from colors_of_meaning.interface.cli.compress import main, CompressArgs
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestCompressCLI:
    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.compress.pickle")
    @patch("colors_of_meaning.interface.cli.compress.CompressDocumentUseCase")
    @patch("builtins.print")
    def test_should_execute_compress_workflow(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        tmp_path: Path,
    ) -> None:
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), color_sequence=[0, 1, 1])
        doc2 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), color_sequence=[0, 0, 1])
        mock_pickle.load.return_value = [doc1, doc2]

        mock_use_case = Mock()
        batch_results = {
            "total_bits": 1000,
            "total_tokens": 100,
            "average_bits_per_token": 10.0,
            "individual_results": [
                {
                    "num_tokens": 3,
                    "total_bits": 50,
                    "bits_per_token": 16.67,
                    "compression_ratio": 4.8,
                },
                {
                    "num_tokens": 3,
                    "total_bits": 50,
                    "bits_per_token": 16.67,
                    "compression_ratio": 4.8,
                },
            ],
        }
        mock_use_case.execute_batch.return_value = batch_results
        mock_use_case_class.return_value = mock_use_case

        config_path = tmp_path / "config.yaml"
        encoded_path = tmp_path / "encoded.pkl"

        args = CompressArgs(
            config=str(config_path),
            encoded_documents=str(encoded_path),
            method="vq",
        )

        main(args)

        mock_use_case.execute_batch.assert_called_once_with([doc1, doc2])
        assert mock_print.call_count > 0

    @patch("builtins.open", new_callable=mock_open)
    @patch("colors_of_meaning.interface.cli.compress.pickle")
    @patch("colors_of_meaning.interface.cli.compress.CompressDocumentUseCase")
    @patch("builtins.print")
    def test_should_handle_large_number_of_documents(
        self,
        mock_print: Mock,
        mock_use_case_class: Mock,
        mock_pickle: Mock,
        mock_file: Mock,
        tmp_path: Path,
    ) -> None:
        documents = [
            ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), color_sequence=[0, 1]) for _ in range(10)
        ]
        mock_pickle.load.return_value = documents

        mock_use_case = Mock()
        individual_results = [
            {
                "num_tokens": 2,
                "total_bits": 20,
                "bits_per_token": 10.0,
                "compression_ratio": 8.0,
            }
            for _ in range(10)
        ]
        batch_results = {
            "total_bits": 200,
            "total_tokens": 20,
            "average_bits_per_token": 10.0,
            "individual_results": individual_results,
        }
        mock_use_case.execute_batch.return_value = batch_results
        mock_use_case_class.return_value = mock_use_case

        args = CompressArgs(
            encoded_documents=str(tmp_path / "encoded.pkl"),
        )

        main(args)

        mock_use_case.execute_batch.assert_called_once()
