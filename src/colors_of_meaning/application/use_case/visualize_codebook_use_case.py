from colors_of_meaning.domain.repository.color_codebook_repository import (
    ColorCodebookRepository,
)
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer


class VisualizeCodebookUseCase:
    def __init__(
        self,
        codebook_repository: ColorCodebookRepository,
        figure_renderer: FigureRenderer,
    ) -> None:
        self.codebook_repository = codebook_repository
        self.figure_renderer = figure_renderer

    def execute(self, codebook_name: str, output_path: str) -> None:
        codebook = self.codebook_repository.load(codebook_name)
        if codebook is None:
            raise FileNotFoundError(f"Codebook not found: {codebook_name}")
        self.figure_renderer.render_codebook_palette(codebook, output_path)
