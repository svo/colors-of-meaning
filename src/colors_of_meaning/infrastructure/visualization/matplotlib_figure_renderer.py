import math
import os
from typing import List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from sklearn.manifold import TSNE  # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.figure_renderer import FigureRenderer
from colors_of_meaning.shared.lab_utils import lab_to_rgb

matplotlib.use("Agg")


class MatplotlibFigureRenderer(FigureRenderer):
    def render_codebook_palette(self, codebook: ColorCodebook, output_path: str) -> None:
        grid_size = int(math.ceil(math.sqrt(codebook.num_bins)))
        image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        for i in range(codebook.num_bins):
            row = i // grid_size
            col = i % grid_size
            rgb = lab_to_rgb(codebook.colors[i])
            image[row, col] = rgb

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(image, interpolation="nearest")
        ax.set_title(f"Color Codebook ({codebook.num_bins} colors)")
        ax.axis("off")

        self._save_figure(fig, output_path)

    def render_document_histograms(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
        samples_per_class: int = 2,
    ) -> None:
        selected_indices = self._select_samples_per_class(labels, len(label_names), samples_per_class)

        num_plots = len(selected_indices)
        cols = min(num_plots, 2)
        rows = int(math.ceil(num_plots / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows))
        if num_plots == 1:
            axes = np.array([axes])
        axes = np.atleast_2d(axes)

        for plot_idx, doc_idx in enumerate(selected_indices):
            row = plot_idx // cols
            col = plot_idx % cols
            ax = axes[row, col]
            document = documents[doc_idx]
            label = labels[doc_idx]

            nonzero_mask = document.histogram > 0
            nonzero_bins = np.where(nonzero_mask)[0]
            nonzero_values = document.histogram[nonzero_mask]

            ax.bar(range(len(nonzero_bins)), nonzero_values, width=1.0)
            ax.set_title(f"{label_names[label]} (doc {doc_idx})")
            ax.set_xlabel("Color bin")
            ax.set_ylabel("Frequency")

        for plot_idx in range(num_plots, rows * cols):
            row = plot_idx // cols
            col = plot_idx % cols
            axes[row, col].axis("off")

        fig.suptitle("Document Color Histograms by Class")
        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_tsne_projection(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        histograms = np.array([doc.histogram for doc in documents])

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        projections = tsne.fit_transform(histograms)

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = sorted(set(labels))
        cmap = get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(unique_labels)))

        for label_idx, label in enumerate(unique_labels):
            mask = np.array(labels) == label
            ax.scatter(
                projections[mask, 0],
                projections[mask, 1],
                c=[colors[label_idx]],
                label=label_names[label],
                alpha=0.6,
                s=20,
            )

        ax.set_title("t-SNE Projection of Color Histograms")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.legend()

        fig.tight_layout()
        self._save_figure(fig, output_path)

    def render_confusion_matrix(
        self,
        y_true: List[int],
        y_pred: List[int],
        label_names: List[str],
        output_path: str,
    ) -> None:
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(cm, interpolation="nearest", cmap=get_cmap("Blues"))
        ax.figure.colorbar(im, ax=ax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=label_names,
            yticklabels=label_names,
            ylabel="True label",
            xlabel="Predicted label",
            title="Confusion Matrix",
        )

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                )

        fig.tight_layout()
        self._save_figure(fig, output_path)

    @staticmethod
    def _select_samples_per_class(labels: List[int], num_classes: int, samples_per_class: int) -> List[int]:
        selected: List[int] = []
        for class_idx in range(num_classes):
            class_indices = [i for i, label in enumerate(labels) if label == class_idx]
            selected.extend(class_indices[:samples_per_class])
        return selected

    @staticmethod
    def _save_figure(fig: plt.Figure, output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
