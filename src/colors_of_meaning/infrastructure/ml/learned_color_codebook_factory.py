import logging
import uuid
from typing import List

import numpy as np
import numpy.typing as npt
from sklearn.cluster import MiniBatchKMeans  # type: ignore[import-untyped]

from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor
from colors_of_meaning.domain.service.color_codebook_factory import ColorCodebookFactory
from colors_of_meaning.domain.service.color_mapper import ColorMapper

logger = logging.getLogger(__name__)

LAB_LOWER_BOUNDS = np.array([0.0, -128.0, -128.0], dtype=np.float64)
LAB_UPPER_BOUNDS = np.array([100.0, 127.0, 127.0], dtype=np.float64)


class LearnedColorCodebookFactory(ColorCodebookFactory):
    def __init__(self, color_mapper: ColorMapper) -> None:
        self.color_mapper = color_mapper

    def build(self, embeddings: npt.NDArray, num_bins: int, seed: int) -> ColorCodebook:
        projected_points = self._project_to_lab_points(embeddings)
        cluster_count = self._resolve_cluster_count(projected_points, num_bins)
        estimator = self._fit_estimator(projected_points, cluster_count, seed)
        palette = self._build_palette(estimator.cluster_centers_, num_bins)
        self._log_fit(num_bins, len(projected_points), float(estimator.inertia_))
        return ColorCodebook(colors=palette, num_bins=num_bins)

    def _project_to_lab_points(self, embeddings: npt.NDArray) -> npt.NDArray:
        lab_colors = self.color_mapper.embed_batch_to_lab(embeddings)
        if not lab_colors:
            raise ValueError("Cannot build a learned codebook from empty embeddings")
        return np.array([[color.l, color.a, color.b] for color in lab_colors], dtype=np.float32)

    def _resolve_cluster_count(self, projected_points: npt.NDArray, num_bins: int) -> int:
        distinct_point_count = len(np.unique(projected_points, axis=0))
        return min(num_bins, distinct_point_count)

    def _fit_estimator(self, projected_points: npt.NDArray, cluster_count: int, seed: int) -> MiniBatchKMeans:
        estimator = MiniBatchKMeans(
            n_clusters=cluster_count,
            random_state=seed,
            n_init=1,
            batch_size=min(256, len(projected_points)),
        )
        estimator.fit(projected_points)
        return estimator

    def _build_palette(self, cluster_centers: npt.NDArray, num_bins: int) -> List[LabColor]:
        clamped_centers = np.clip(np.asarray(cluster_centers, dtype=np.float64), LAB_LOWER_BOUNDS, LAB_UPPER_BOUNDS)
        centroid_colors = [LabColor(l=float(row[0]), a=float(row[1]), b=float(row[2])) for row in clamped_centers]
        return [centroid_colors[index % len(centroid_colors)] for index in range(num_bins)]

    def _log_fit(self, num_bins: int, num_projected_colors: int, inertia: float) -> None:
        logger.info(
            "Fitted learned color codebook",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "codebook_mode": "learned",
                "num_bins": num_bins,
                "num_projected_colors": num_projected_colors,
                "inertia": inertia,
            },
        )
