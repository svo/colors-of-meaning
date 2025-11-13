import pytest
import numpy as np
from colors_of_meaning.domain.model.colored_document import ColoredDocument


class TestColoredDocument:
    def test_should_create_colored_document_with_normalized_histogram(self) -> None:
        histogram = np.array([0.5, 0.3, 0.2], dtype=np.float64)

        doc = ColoredDocument(histogram=histogram)

        assert doc.num_bins == 3
        assert np.allclose(doc.histogram.sum(), 1.0)

    def test_should_raise_error_when_histogram_is_not_normalized(self) -> None:
        histogram = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        with pytest.raises(ValueError, match="histogram must be normalized"):
            ColoredDocument(histogram=histogram)

    def test_should_raise_error_when_histogram_has_negative_values(self) -> None:
        histogram = np.array([0.5, -0.3, 0.8], dtype=np.float64)
        histogram = histogram / histogram.sum()

        with pytest.raises(ValueError, match="histogram values must be non-negative"):
            ColoredDocument(histogram=histogram)

    def test_should_create_from_color_sequence(self) -> None:
        color_sequence = [0, 1, 1, 2, 0]

        doc = ColoredDocument.from_color_sequence(color_sequence, num_bins=3)

        assert doc.num_bins == 3
        assert np.isclose(doc.histogram[0], 2.0 / 5.0)
        assert np.isclose(doc.histogram[1], 2.0 / 5.0)
        assert np.isclose(doc.histogram[2], 1.0 / 5.0)

    def test_should_compute_variance_from_color_sequence(self) -> None:
        color_sequence = [0, 1, 2, 3, 4]
        histogram = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=color_sequence)

        variance = doc.compute_variance()

        assert variance == 2.0

    def test_should_return_zero_variance_when_no_sequence(self) -> None:
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram)

        variance = doc.compute_variance()

        assert variance == 0.0

    def test_should_normalize_histogram(self) -> None:
        histogram = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        histogram = histogram / histogram.sum()
        doc = ColoredDocument(histogram=histogram)

        normalized = doc.normalize()

        assert np.isclose(normalized.histogram.sum(), 1.0)

    def test_should_raise_error_when_histogram_is_not_array(self) -> None:
        with pytest.raises(TypeError, match="histogram must be a numpy array"):
            ColoredDocument(histogram=[0.5, 0.5])  # type: ignore

    def test_should_raise_error_when_histogram_is_not_1d(self) -> None:
        histogram = np.array([[0.5, 0.5]], dtype=np.float64)

        with pytest.raises(ValueError, match="histogram must be 1D"):
            ColoredDocument(histogram=histogram)

    def test_should_normalize_zero_histogram(self) -> None:
        histogram = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        histogram = histogram / histogram.sum()
        doc = ColoredDocument(histogram=histogram)

        normalized = doc.normalize()

        assert np.allclose(normalized.histogram, doc.histogram)

    def test_should_return_zero_autocorrelation_when_no_sequence(self) -> None:
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram)

        autocorr = doc.compute_autocorrelation(lag=1)

        assert autocorr == 0.0

    def test_should_return_zero_autocorrelation_when_sequence_too_short(self) -> None:
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=[0])

        autocorr = doc.compute_autocorrelation(lag=1)

        assert autocorr == 0.0

    def test_should_compute_autocorrelation_with_valid_sequence(self) -> None:
        color_sequence = [0, 1, 0, 1, 0, 1]
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=color_sequence)

        autocorr = doc.compute_autocorrelation(lag=1)

        assert isinstance(autocorr, float)

    def test_should_return_zero_autocorrelation_when_variance_is_zero(self) -> None:
        color_sequence = [0, 0, 0, 0]
        histogram = np.array([1.0], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=color_sequence)

        autocorr = doc.compute_autocorrelation(lag=1)

        assert autocorr == 0.0

    def test_should_raise_error_when_color_sequence_is_empty(self) -> None:
        with pytest.raises(ValueError, match="color_sequence cannot be empty"):
            ColoredDocument.from_color_sequence([], num_bins=3)

    def test_should_raise_error_when_color_bin_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="color_bin .* out of range"):
            ColoredDocument.from_color_sequence([0, 1, 5], num_bins=3)

    def test_should_create_from_color_sequence_with_document_id(self) -> None:
        color_sequence = [0, 1, 1]
        doc_id = "test_doc"

        doc = ColoredDocument.from_color_sequence(color_sequence, num_bins=2, document_id=doc_id)

        assert doc.document_id == doc_id
        assert doc.color_sequence == color_sequence

    def test_should_preserve_color_sequence_in_normalize(self) -> None:
        color_sequence = [0, 1, 1]
        histogram = np.array([0.33, 0.67], dtype=np.float64)
        doc_id = "test_doc"
        doc = ColoredDocument(histogram=histogram, color_sequence=color_sequence, document_id=doc_id)

        normalized = doc.normalize()

        assert normalized.color_sequence == color_sequence
        assert normalized.document_id == doc_id

    def test_should_return_zero_variance_when_empty_sequence(self) -> None:
        histogram = np.array([0.5, 0.5], dtype=np.float64)
        doc = ColoredDocument(histogram=histogram, color_sequence=[])

        variance = doc.compute_variance()

        assert variance == 0.0
