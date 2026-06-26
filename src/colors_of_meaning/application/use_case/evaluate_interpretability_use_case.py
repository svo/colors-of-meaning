import logging
import uuid
from typing import List, Optional, Protocol, Sequence, Tuple

import numpy.typing as npt

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample
from colors_of_meaning.domain.model.interpretability_report import (
    InterpretabilityReport,
    InterpretabilityScores,
    InterpretabilityThresholds,
)
from colors_of_meaning.domain.service.color_mapper import ColorMapper
from colors_of_meaning.domain.service.concreteness_lexicon import ConcretenessLexicon
from colors_of_meaning.domain.service.interpretability_evaluator import (
    InterpretabilityEvaluator,
)

logger = logging.getLogger(__name__)

DocumentSignals = Tuple[List[int], List[float], List[float]]


class DocumentEmbedder(Protocol):
    def encode_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = False) -> npt.NDArray:
        raise NotImplementedError


class EvaluateInterpretabilityUseCase:
    def __init__(
        self,
        embedding_adapter: DocumentEmbedder,
        structured_mapper: ColorMapper,
        control_mapper: ColorMapper,
        interpretability_evaluator: InterpretabilityEvaluator,
        concreteness_lexicon: ConcretenessLexicon,
        thresholds: Optional[InterpretabilityThresholds] = None,
    ) -> None:
        self._embedding_adapter = embedding_adapter
        self._structured_mapper = structured_mapper
        self._control_mapper = control_mapper
        self._evaluator = interpretability_evaluator
        self._concreteness_lexicon = concreteness_lexicon
        self._thresholds = thresholds or InterpretabilityThresholds()

    def execute(self, samples: Sequence[EvaluationSample]) -> InterpretabilityReport:
        embeddings = self._embedding_adapter.encode_batch([sample.text for sample in samples])
        signals = self._gather_signals(samples)
        structured_scores = self._score_mapper(self._structured_mapper, embeddings, signals)
        control_scores = self._score_mapper(self._control_mapper, embeddings, signals)
        report = InterpretabilityReport(
            structured=structured_scores, control=control_scores, thresholds=self._thresholds
        )
        self._log_report(report, len(samples))
        return report

    def _gather_signals(self, samples: Sequence[EvaluationSample]) -> DocumentSignals:
        topics = [sample.label for sample in samples]
        sentiments = [float(sample.label) for sample in samples]
        concreteness = [self._concreteness_lexicon.score(sample.text) for sample in samples]
        return topics, sentiments, concreteness

    def _score_mapper(
        self, mapper: ColorMapper, embeddings: npt.NDArray, signals: DocumentSignals
    ) -> InterpretabilityScores:
        topics, sentiments, concreteness = signals
        document_colors = mapper.embed_batch_to_lab(embeddings)
        return self._evaluator.evaluate(document_colors, topics, sentiments, concreteness)

    def _log_report(self, report: InterpretabilityReport, sample_count: int) -> None:
        logger.info(
            "Evaluated structured-mapper interpretability against control",
            extra={
                "correlation_id": str(uuid.uuid4()),
                "hue_topic_score": report.structured.hue_topic_score,
                "lightness_sentiment_score": report.structured.lightness_sentiment_score,
                "chroma_concreteness_score": report.structured.chroma_concreteness_score,
                "control_hue_topic_score": report.control.hue_topic_score,
                "control_lightness_sentiment_score": report.control.lightness_sentiment_score,
                "control_chroma_concreteness_score": report.control.chroma_concreteness_score,
                "falsified_axes": report.falsified_axes,
                "sample_count": sample_count,
            },
        )
