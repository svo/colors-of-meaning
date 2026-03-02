import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import numpy as np

from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer
from colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer import (
    MatplotlibFigureRenderer,
)


class TestMatplotlibFigureRendererInterface:
    def test_should_implement_figure_renderer_interface(self) -> None:
        renderer = MatplotlibFigureRenderer()

        assert isinstance(renderer, FigureRenderer)


class TestRenderCodebookPalette:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_create_figure_for_codebook(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.imshow.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_set_title_with_codebook_size(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.set_title.assert_called_once_with("Color Codebook (4 colors)")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_hide_axes_for_codebook_palette(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_ax.axis.assert_called_once_with("off")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_save_codebook_palette_to_file(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.lab_to_rgb")
    def test_should_close_figure_after_saving(
        self,
        mock_lab_to_rgb: Mock,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_lab_to_rgb.return_value = (128, 128, 128)
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)

        colors = [LabColor(l=50.0, a=0.0, b=0.0) for _ in range(4)]
        codebook = ColorCodebook(colors=colors, num_bins=4)

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "palette.png")
        renderer.render_codebook_palette(codebook, output_path)

        mock_plt.close.assert_called_once_with(mock_fig)


class TestRenderDocumentHistograms:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_subplots_for_document_histograms(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_plt.subplots.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_handle_single_document_histogram(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_single_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_single_ax)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [ColoredDocument(histogram=histogram, document_id="doc_0")]
        labels = [0]
        label_names = ["Class A"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.suptitle.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_hide_unused_axes_in_grid(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_unused_ax = Mock()
        mock_axes = np.array([[Mock(), Mock()], [Mock(), mock_unused_ax]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
            ColoredDocument(histogram=histogram, document_id="doc_2"),
        ]
        labels = [0, 1, 2]
        label_names = ["Class A", "Class B", "Class C"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_unused_ax.axis.assert_called_once_with("off")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_set_suptitle_for_histograms(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.suptitle.assert_called_once_with("Document Color Histograms by Class")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_histogram_figure_to_file(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_axes = np.array([[Mock(), Mock()]])
        mock_plt.subplots.return_value = (mock_fig, mock_axes)

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram, document_id="doc_0"),
            ColoredDocument(histogram=histogram, document_id="doc_1"),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "histograms.png")
        renderer.render_document_histograms(documents, labels, label_names, output_path, samples_per_class=1)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestRenderTsneProjection:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_run_tsne_on_histograms(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_get_cmap.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_tsne_class.assert_called_once_with(n_components=2, perplexity=30, random_state=42)

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_scatter_plot_for_each_class(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_get_cmap.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        assert mock_ax.scatter.call_count == 2

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_set_title_and_labels_for_projection(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_get_cmap.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_ax.set_title.assert_called_once_with("t-SNE Projection of Color Histograms")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.TSNE")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_projection_figure_to_file(
        self,
        mock_plt: Mock,
        mock_tsne_class: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_cmap = MagicMock()
        mock_cmap.return_value = np.array([[1, 0, 0, 1], [0, 1, 0, 1]])
        mock_get_cmap.return_value = mock_cmap

        mock_tsne = Mock()
        mock_tsne.fit_transform.return_value = np.array([[1.0, 2.0], [3.0, 4.0]])
        mock_tsne_class.return_value = mock_tsne

        histogram = np.array([0.5, 0.3, 0.2])
        documents = [
            ColoredDocument(histogram=histogram),
            ColoredDocument(histogram=histogram),
        ]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "projection.png")
        renderer.render_tsne_projection(documents, labels, label_names, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestRenderConfusionMatrix:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_compute_confusion_matrix(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_cm_fn.assert_called_once_with([0, 0, 1, 1], [0, 1, 0, 1])

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_display_confusion_matrix_as_heatmap(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_ax.imshow.assert_called_once()

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_annotate_confusion_matrix_cells(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        assert mock_ax.text.call_count == 4

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.get_cmap")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.confusion_matrix")
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_confusion_matrix_figure_to_file(
        self,
        mock_plt: Mock,
        mock_cm_fn: Mock,
        mock_get_cmap: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.figure = mock_fig
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.setp = Mock()

        cm_array = np.array([[10, 2], [3, 15]])
        mock_cm_fn.return_value = cm_array

        renderer = MatplotlibFigureRenderer()
        output_path = str(tmp_path / "confusion.png")
        renderer.render_confusion_matrix([0, 0, 1, 1], [0, 1, 0, 1], ["A", "B"], output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")


class TestSelectSamplesPerClass:
    def test_should_select_correct_number_of_samples_per_class(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 3, 2)

        assert len(selected) == 6

    def test_should_select_from_each_class(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 3, 1)

        assert selected == [0, 2, 4]

    def test_should_handle_fewer_samples_than_requested(self) -> None:
        labels = [0, 1]

        selected = MatplotlibFigureRenderer._select_samples_per_class(labels, 2, 3)

        assert len(selected) == 2


class TestSaveFigure:
    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_create_output_directory_if_not_exists(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        output_path = str(tmp_path / "subdir" / "figure.png")

        MatplotlibFigureRenderer._save_figure(mock_fig, output_path)

        assert os.path.isdir(str(tmp_path / "subdir"))

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_save_with_correct_dpi(
        self,
        mock_plt: Mock,
        tmp_path: Path,
    ) -> None:
        mock_fig = Mock()
        output_path = str(tmp_path / "figure.png")

        MatplotlibFigureRenderer._save_figure(mock_fig, output_path)

        mock_fig.savefig.assert_called_once_with(output_path, dpi=150, bbox_inches="tight")

    @patch("colors_of_meaning.infrastructure.visualization.matplotlib_figure_renderer.plt")
    def test_should_handle_filename_without_directory(
        self,
        mock_plt: Mock,
    ) -> None:
        mock_fig = Mock()

        MatplotlibFigureRenderer._save_figure(mock_fig, "figure.png")

        mock_fig.savefig.assert_called_once_with("figure.png", dpi=150, bbox_inches="tight")
