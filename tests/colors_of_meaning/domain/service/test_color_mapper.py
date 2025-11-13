from unittest.mock import Mock
import numpy as np

from colors_of_meaning.domain.service.color_mapper import QuantizedColorMapper
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook


class TestQuantizedColorMapper:
    def test_should_convert_embedding_to_bin(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_to_lab.return_value = LabColor(l=50.0, a=10.0, b=-20.0)

        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=10.0, b=-20.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        quantized_mapper = QuantizedColorMapper(mock_mapper, codebook)
        embedding = np.array([1.0, 2.0, 3.0])

        bin_index = quantized_mapper.embed_to_bin(embedding)

        assert bin_index == 1
        mock_mapper.embed_to_lab.assert_called_once()

    def test_should_convert_batch_embeddings_to_bins(self) -> None:
        mock_mapper = Mock()
        mock_mapper.embed_batch_to_lab.return_value = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=10.0, b=-20.0),
        ]

        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=10.0, b=-20.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        quantized_mapper = QuantizedColorMapper(mock_mapper, codebook)
        embeddings = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        bin_indices = quantized_mapper.embed_batch_to_bins(embeddings)

        assert bin_indices == [0, 1]
        mock_mapper.embed_batch_to_lab.assert_called_once()
