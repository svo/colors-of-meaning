from unittest.mock import Mock

from colors_of_meaning.application.use_case.visualize_documents_use_case import (
    VisualizeDocumentsUseCase,
)


class TestVisualizeDocumentsUseCaseHistograms:
    def test_should_delegate_histogram_rendering_to_figure_renderer(self) -> None:
        mock_renderer = Mock()
        documents = [Mock(), Mock()]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_histograms(documents, labels, label_names, "/output/histograms.png")

        mock_renderer.render_document_histograms.assert_called_once_with(
            documents, labels, label_names, "/output/histograms.png", 2
        )

    def test_should_pass_custom_samples_per_class(self) -> None:
        mock_renderer = Mock()
        documents = [Mock(), Mock()]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_histograms(documents, labels, label_names, "/output/histograms.png", samples_per_class=3)

        mock_renderer.render_document_histograms.assert_called_once_with(
            documents, labels, label_names, "/output/histograms.png", 3
        )


class TestVisualizeDocumentsUseCaseProjection:
    def test_should_delegate_projection_rendering_to_figure_renderer(self) -> None:
        mock_renderer = Mock()
        documents = [Mock(), Mock()]
        labels = [0, 1]
        label_names = ["Class A", "Class B"]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_projection(documents, labels, label_names, "/output/projection.png")

        mock_renderer.render_tsne_projection.assert_called_once_with(
            documents, labels, label_names, "/output/projection.png"
        )


class TestVisualizeDocumentsUseCaseConfusionMatrix:
    def test_should_delegate_confusion_matrix_rendering_to_figure_renderer(self) -> None:
        mock_renderer = Mock()
        y_true = [0, 0, 1, 1]
        y_pred = [0, 1, 0, 1]
        label_names = ["Class A", "Class B"]

        use_case = VisualizeDocumentsUseCase(mock_renderer)
        use_case.execute_confusion_matrix(y_true, y_pred, label_names, "/output/confusion.png")

        mock_renderer.render_confusion_matrix.assert_called_once_with(
            y_true, y_pred, label_names, "/output/confusion.png"
        )
