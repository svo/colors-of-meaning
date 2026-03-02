import pytest

from colors_of_meaning.domain.service.figure_renderer import FigureRenderer


class TestFigureRenderer:
    def test_should_not_instantiate_abstract_class(self) -> None:
        with pytest.raises(TypeError):
            FigureRenderer()  # type: ignore

    def test_should_define_render_codebook_palette_method(self) -> None:
        assert hasattr(FigureRenderer, "render_codebook_palette")

    def test_should_define_render_document_histograms_method(self) -> None:
        assert hasattr(FigureRenderer, "render_document_histograms")

    def test_should_define_render_tsne_projection_method(self) -> None:
        assert hasattr(FigureRenderer, "render_tsne_projection")

    def test_should_define_render_confusion_matrix_method(self) -> None:
        assert hasattr(FigureRenderer, "render_confusion_matrix")

    def test_should_allow_concrete_implementation(self) -> None:
        class ConcreteFigureRenderer(FigureRenderer):
            def render_codebook_palette(self, codebook, output_path):  # type: ignore
                pass

            def render_document_histograms(self, documents, labels, label_names, output_path, samples_per_class=2):  # type: ignore  # noqa: E501
                pass

            def render_tsne_projection(self, documents, labels, label_names, output_path):  # type: ignore
                pass

            def render_confusion_matrix(self, y_true, y_pred, label_names, output_path):  # type: ignore
                pass

        renderer = ConcreteFigureRenderer()

        assert isinstance(renderer, FigureRenderer)
