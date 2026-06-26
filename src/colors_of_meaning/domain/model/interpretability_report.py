from dataclasses import dataclass, field
from typing import Dict, List, Tuple

HUE_TOPIC_AXIS = "hue_topic"
LIGHTNESS_SENTIMENT_AXIS = "lightness_sentiment"
CHROMA_CONCRETENESS_AXIS = "chroma_concreteness"
INTERPRETABILITY_AXES: Tuple[str, str, str] = (
    HUE_TOPIC_AXIS,
    LIGHTNESS_SENTIMENT_AXIS,
    CHROMA_CONCRETENESS_AXIS,
)


@dataclass(frozen=True)
class InterpretabilityScores:
    hue_topic_score: float
    lightness_sentiment_score: float
    chroma_concreteness_score: float

    def __post_init__(self) -> None:
        self._validate_unit_interval("hue_topic_score", self.hue_topic_score)
        self._validate_correlation("lightness_sentiment_score", self.lightness_sentiment_score)
        self._validate_correlation("chroma_concreteness_score", self.chroma_concreteness_score)

    @property
    def axis_scores(self) -> Dict[str, float]:
        return {
            HUE_TOPIC_AXIS: self.hue_topic_score,
            LIGHTNESS_SENTIMENT_AXIS: self.lightness_sentiment_score,
            CHROMA_CONCRETENESS_AXIS: self.chroma_concreteness_score,
        }

    @staticmethod
    def _validate_unit_interval(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1, got {value}")

    @staticmethod
    def _validate_correlation(name: str, value: float) -> None:
        if not -1.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between -1 and 1, got {value}")


@dataclass(frozen=True)
class InterpretabilityThresholds:
    hue_topic_margin: float = 0.05
    lightness_sentiment_margin: float = 0.05
    chroma_concreteness_margin: float = 0.05

    def __post_init__(self) -> None:
        for name, value in self.axis_margins.items():
            if value < 0.0:
                raise ValueError(f"{name} threshold must be non-negative, got {value}")

    @property
    def axis_margins(self) -> Dict[str, float]:
        return {
            HUE_TOPIC_AXIS: self.hue_topic_margin,
            LIGHTNESS_SENTIMENT_AXIS: self.lightness_sentiment_margin,
            CHROMA_CONCRETENESS_AXIS: self.chroma_concreteness_margin,
        }


@dataclass(frozen=True)
class InterpretabilityReport:
    structured: InterpretabilityScores
    control: InterpretabilityScores
    thresholds: InterpretabilityThresholds = field(default_factory=InterpretabilityThresholds)

    @property
    def margins(self) -> Dict[str, float]:
        structured = self.structured.axis_scores
        control = self.control.axis_scores
        return {axis: structured[axis] - control[axis] for axis in INTERPRETABILITY_AXES}

    @property
    def falsified_axes(self) -> List[str]:
        margins = self.margins
        thresholds = self.thresholds.axis_margins
        return [axis for axis in INTERPRETABILITY_AXES if margins[axis] < thresholds[axis]]

    @property
    def is_validated(self) -> bool:
        return not self.falsified_axes
