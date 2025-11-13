from unittest.mock import Mock
import numpy as np

from colors_of_meaning.application.use_case.compare_documents_use_case import CompareDocumentsUseCase
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestCompareDocumentsUseCase:
    def test_should_compute_distance_between_documents(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.return_value = 0.5

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        distance = use_case.execute(doc1, doc2)

        assert distance == 0.5
        mock_distance_calculator.compute_distance.assert_called_once_with(doc1, doc2)

    def test_should_compute_pairwise_distances(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.side_effect = [0.1, 0.2, 0.3]

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="doc1")
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64), document_id="doc2")
        doc3 = ColoredDocument(histogram=np.array([0.4, 0.6], dtype=np.float64), document_id="doc3")

        results = use_case.execute_pairwise([doc1, doc2, doc3])

        assert len(results) == 3
        assert results[0] == ("doc1", "doc2", 0.1)
        assert results[1] == ("doc1", "doc3", 0.2)
        assert results[2] == ("doc2", "doc3", 0.3)

    def test_should_compute_pairwise_distances_with_default_ids(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.return_value = 0.1

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))
        doc2 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        results = use_case.execute_pairwise([doc1, doc2])

        assert len(results) == 1
        assert results[0] == ("doc_0", "doc_1", 0.1)

    def test_should_find_nearest_neighbors(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.side_effect = [0.5, 0.1, 0.3]

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        query_doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="query")
        doc1 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64), document_id="doc1")
        doc2 = ColoredDocument(histogram=np.array([0.4, 0.6], dtype=np.float64), document_id="doc2")
        doc3 = ColoredDocument(histogram=np.array([0.6, 0.4], dtype=np.float64), document_id="doc3")

        neighbors = use_case.find_nearest_neighbors(query_doc, [doc1, doc2, doc3], k=2)

        assert len(neighbors) == 2
        assert neighbors[0] == ("doc2", 0.1)
        assert neighbors[1] == ("doc3", 0.3)

    def test_should_skip_query_document_in_corpus(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.return_value = 0.5

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        query_doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="query")
        doc1 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64), document_id="query")
        doc2 = ColoredDocument(histogram=np.array([0.4, 0.6], dtype=np.float64), document_id="doc2")

        neighbors = use_case.find_nearest_neighbors(query_doc, [doc1, doc2], k=5)

        assert len(neighbors) == 1
        assert neighbors[0][0] == "doc2"

    def test_should_handle_missing_document_id(self) -> None:
        mock_distance_calculator = Mock()
        mock_distance_calculator.compute_distance.return_value = 0.5

        use_case = CompareDocumentsUseCase(mock_distance_calculator)
        query_doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), document_id="query")
        doc1 = ColoredDocument(histogram=np.array([0.3, 0.7], dtype=np.float64))

        neighbors = use_case.find_nearest_neighbors(query_doc, [doc1], k=1)

        assert len(neighbors) == 1
        assert neighbors[0][0] == "unknown"
