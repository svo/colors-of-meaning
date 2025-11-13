from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter import IMDBDatasetAdapter


class TestIMDBDatasetAdapter:
    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_load_train_samples(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [
            {"text": "Great movie!", "label": 1},
            {"text": "Terrible movie.", "label": 0},
        ]
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        result = adapter.get_samples("train")

        assert len(result) == 2

    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_limit_samples_with_max_samples(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [
            {"text": "Great movie!", "label": 1},
            {"text": "Terrible movie.", "label": 0},
        ]
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        result = adapter.get_samples("train", max_samples=1)

        assert len(result) == 1

    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_parse_text_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "Great movie!", "label": 1}]
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].text == "Great movie!"

    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_parse_label_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "Great movie!", "label": 1}]
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].label == 1

    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_set_split_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "Great movie!", "label": 1}]
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        result = adapter.get_samples("test")

        assert result[0].split == "test"

    def test_should_return_label_names(self) -> None:
        adapter = IMDBDatasetAdapter()

        result = adapter.get_label_names()

        assert len(result) == 2

    def test_should_return_correct_label_names(self) -> None:
        adapter = IMDBDatasetAdapter()

        result = adapter.get_label_names()

        assert result == ["negative", "positive"]

    def test_should_return_num_classes(self) -> None:
        adapter = IMDBDatasetAdapter()

        result = adapter.get_num_classes()

        assert result == 2

    @patch("colors_of_meaning.infrastructure.dataset.imdb_dataset_adapter.load_dataset")
    def test_should_call_load_dataset_with_correct_parameters(self, mock_load_dataset: Mock) -> None:
        mock_dataset = []
        mock_load_dataset.return_value = mock_dataset

        adapter = IMDBDatasetAdapter()
        adapter.get_samples("train")

        mock_load_dataset.assert_called_once_with("stanfordnlp/imdb", split="train")
