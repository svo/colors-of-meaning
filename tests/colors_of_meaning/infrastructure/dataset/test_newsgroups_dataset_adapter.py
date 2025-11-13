from unittest.mock import Mock, patch
from types import SimpleNamespace

from colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter import NewsgroupsDatasetAdapter


class TestNewsgroupsDatasetAdapter:
    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_load_train_samples(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=["Article about space", "Article about religion"],
            target=[0, 1],
            target_names=["sci.space", "soc.religion.christian"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_samples("train")

        assert len(result) == 2

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_limit_samples_with_max_samples(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=["Article about space", "Article about religion"],
            target=[0, 1],
            target_names=["sci.space", "soc.religion.christian"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_samples("train", max_samples=1)

        assert len(result) == 1

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_parse_text_correctly(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=["Article about space"],
            target=[0],
            target_names=["sci.space"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].text == "Article about space"

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_parse_label_correctly(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=["Article about space"],
            target=[5],
            target_names=["sci.space"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_samples("train")

        assert result[0].label == 5

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_set_split_correctly(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=["Article about space"],
            target=[0],
            target_names=["sci.space"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_samples("test")

        assert result[0].split == "test"

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_return_label_names(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=[],
            target=[],
            target_names=["sci.space", "soc.religion.christian"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_label_names()

        assert len(result) == 2

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_return_correct_label_names(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=[],
            target=[],
            target_names=["sci.space", "soc.religion.christian"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_label_names()

        assert result == ["sci.space", "soc.religion.christian"]

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_return_num_classes(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=[],
            target=[],
            target_names=["sci.space", "soc.religion.christian"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        result = adapter.get_num_classes()

        assert result == 2

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_call_fetch_with_train_subset(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=[],
            target=[],
            target_names=["sci.space"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        adapter.get_samples("train")

        mock_fetch.assert_called_with(subset="train", remove=("headers", "footers", "quotes"))

    @patch("colors_of_meaning.infrastructure.dataset.newsgroups_dataset_adapter.fetch_20newsgroups")
    def test_should_call_fetch_with_test_subset_when_test_split_requested(self, mock_fetch: Mock) -> None:
        mock_newsgroups = SimpleNamespace(
            data=[],
            target=[],
            target_names=["sci.space"],
        )
        mock_fetch.return_value = mock_newsgroups

        adapter = NewsgroupsDatasetAdapter()
        adapter.get_samples("test")

        mock_fetch.assert_called_with(subset="test", remove=("headers", "footers", "quotes"))
