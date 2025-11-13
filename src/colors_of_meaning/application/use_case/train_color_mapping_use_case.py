import numpy.typing as npt

from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)


class TrainColorMappingUseCase:
    def __init__(
        self,
        color_mapper: ColorMapper,
        codebook_repository: ColorCodebookRepository,
    ) -> None:
        self.color_mapper = color_mapper
        self.codebook_repository = codebook_repository

    def execute(
        self,
        embeddings: npt.NDArray,
        epochs: int,
        learning_rate: float,
        bins_per_dimension: int,
        model_name: str,
        codebook_name: str,
    ) -> None:
        self.color_mapper.train(embeddings=embeddings, epochs=epochs, learning_rate=learning_rate)

        self.color_mapper.save_weights(model_name)

        codebook = ColorCodebook.create_uniform_grid(bins_per_dimension=bins_per_dimension)
        self.codebook_repository.save(codebook, codebook_name)
