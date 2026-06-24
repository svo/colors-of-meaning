import math
import os
from typing import Any, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
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

        perplexity = min(30, len(documents) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        projections = tsne.fit_transform(histograms)

        fig, ax = plt.subplots(figsize=(10, 8))

        unique_labels = sorted(set(labels))
        cmap = matplotlib.colormaps["tab10"]
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
        im = ax.imshow(cm, interpolation="nearest", cmap=matplotlib.colormaps["Blues"])
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

    def render_corpus_signatures(
        self,
        documents: List[ColoredDocument],
        labels: List[int],
        label_names: List[str],
        codebook: ColorCodebook,
        output_path: str,
        top_colors: int = 24,
    ) -> None:
        fig, axes = plt.subplots(len(label_names), 1, figsize=(12, 1.5 * len(label_names)))
        axes = np.atleast_1d(axes)

        for label_index, name in enumerate(label_names):
            histograms = self._corpus_histograms(documents, labels, label_index)
            if not histograms:
                continue
            ax = axes[label_index]
            self._draw_color_signature(ax, np.mean(histograms, axis=0), codebook, top_colors)
            ax.set_ylabel(name, rotation=0, ha="right", va="center")

        fig.suptitle(f"Per-corpus color signature (top {top_colors} colors)")
        fig.tight_layout()
        self._save_figure(fig, output_path)

    @staticmethod
    def _corpus_histograms(
        documents: List[ColoredDocument], labels: List[int], label_index: int
    ) -> List[npt.NDArray[np.float64]]:
        return [documents[i].histogram for i in range(len(documents)) if labels[i] == label_index]

    @staticmethod
    def _draw_color_signature(ax: Any, mean_histogram: Any, codebook: ColorCodebook, top_colors: int) -> None:
        top_bins = np.argsort(mean_histogram)[::-1][:top_colors]
        left = 0.0
        for bin_index in top_bins:
            width = float(mean_histogram[bin_index])
            rgb = np.array(lab_to_rgb(codebook.colors[int(bin_index)]), dtype=float) / 255.0
            ax.barh(0, width, left=left, height=1.0, color=rgb)
            left += width
        ax.set_xlim(0, left)
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_xticks([])

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
