from unittest.mock import Mock, patch

from colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter import AGNewsDatasetAdapter


class TestAGNewsDatasetAdapter:
    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_load_train_samples(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [
            {"text": "World news article", "label": 0},
            {"text": "Sports news article", "label": 1},
        ]
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        result = adapter.get_samples("train")

        assert len(result) == 2

    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_limit_samples_with_max_samples(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [
            {"text": "World news article", "label": 0},
            {"text": "Sports news article", "label": 1},
        ]
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        result = adapter.get_samples("train", max_samples=1)

        assert len(result) == 1

    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_parse_text_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "World news article", "label": 0}]
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].text == "World news article"

    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_parse_label_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "World news article", "label": 2}]
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].label == 2

    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_set_split_correctly(self, mock_load_dataset: Mock) -> None:
        mock_dataset = [{"text": "World news article", "label": 0}]
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        result = adapter.get_samples("test")

        assert result[0].split == "test"

    def test_should_return_label_names(self) -> None:
        adapter = AGNewsDatasetAdapter()

        result = adapter.get_label_names()

        assert len(result) == 4

    def test_should_return_correct_label_names(self) -> None:
        adapter = AGNewsDatasetAdapter()

        result = adapter.get_label_names()

        assert result == ["World", "Sports", "Business", "Sci/Tech"]

    def test_should_return_num_classes(self) -> None:
        adapter = AGNewsDatasetAdapter()

        result = adapter.get_num_classes()

        assert result == 4

    @patch("colors_of_meaning.infrastructure.dataset.ag_news_dataset_adapter.load_dataset")
    def test_should_call_load_dataset_with_correct_parameters(self, mock_load_dataset: Mock) -> None:
        mock_dataset = []
        mock_load_dataset.return_value = mock_dataset

        adapter = AGNewsDatasetAdapter()
        adapter.get_samples("train")

        mock_load_dataset.assert_called_once_with("fancyzhx/ag_news", split="train")
