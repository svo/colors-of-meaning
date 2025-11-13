import numpy as np
import pytest
from typing import Dict, Any

from colors_of_meaning.application.use_case.compress_document_use_case import CompressDocumentUseCase
from colors_of_meaning.domain.model.colored_document import ColoredDocument


def _assert_compression_keys(result: Dict[str, Any]) -> None:
    """Helper to assert compression result has all required keys."""
    assert "palette_bits" in result
    assert "rle_bits" in result
    assert "total_bits" in result


def _assert_compression_metrics(result: Dict[str, Any]) -> None:
    """Helper to assert compression metric values."""
    assert "num_tokens" in result
    assert "bits_per_token" in result
    assert "compression_ratio" in result


def _assert_compression_values(result: Dict[str, Any]) -> None:
    """Helper to assert compression result values are valid."""
    assert result["num_tokens"] == 6
    assert result["total_bits"] > 0


def _assert_batch_compression_keys(result: Dict[str, Any]) -> None:
    """Helper to assert batch compression result keys."""
    assert "total_bits" in result
    assert "total_tokens" in result


def _assert_batch_compression_values(result: Dict[str, Any]) -> None:
    """Helper to assert batch compression values."""
    assert "average_bits_per_token" in result
    assert "individual_results" in result
    assert result["total_tokens"] == 6
    assert len(result["individual_results"]) == 2


class TestCompressDocumentUseCase:
    def test_should_compute_compression_metrics(self) -> None:
        use_case = CompressDocumentUseCase()
        color_sequence = [0, 0, 1, 1, 1, 2]
        histogram = np.array([0.33, 0.5, 0.17], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=color_sequence)

        result = use_case.execute(doc)

        _assert_compression_keys(result)
        _assert_compression_metrics(result)
        _assert_compression_values(result)

    def test_should_raise_error_when_no_color_sequence(self) -> None:
        use_case = CompressDocumentUseCase()
        doc = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64))

        with pytest.raises(ValueError, match="Document must have color_sequence"):
            use_case.execute(doc)

    def test_should_compute_batch_compression(self) -> None:
        use_case = CompressDocumentUseCase()
        doc1 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), color_sequence=[0, 1, 1])
        doc2 = ColoredDocument(histogram=np.array([0.5, 0.5], dtype=np.float64), color_sequence=[0, 0, 1])

        result = use_case.execute_batch([doc1, doc2])

        _assert_batch_compression_keys(result)
        _assert_batch_compression_values(result)

    def test_should_handle_empty_batch_gracefully(self) -> None:
        use_case = CompressDocumentUseCase()

        result = use_case.execute_batch([])

        assert result["total_bits"] == 0
        assert result["total_tokens"] == 0
        assert result["average_bits_per_token"] == 0

    def test_should_compute_palette_bits(self) -> None:
        result = CompressDocumentUseCase._compute_palette_bits(256)

        assert result == 8

    def test_should_compute_rle_bits(self) -> None:
        color_sequence = [0, 0, 0, 1, 1, 2]

        result = CompressDocumentUseCase._compute_rle_bits(color_sequence)

        assert result > 0

    def test_should_compute_compression_ratio(self) -> None:
        result = CompressDocumentUseCase._compute_compression_ratio(100, 10)

        assert result == 8.0

    def test_should_handle_zero_bits_in_compression_ratio(self) -> None:
        result = CompressDocumentUseCase._compute_compression_ratio(0, 10)

        assert result == 0.0
