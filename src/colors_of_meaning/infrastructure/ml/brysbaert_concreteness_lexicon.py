import re
from typing import Dict, List, Optional, Tuple

from colors_of_meaning.domain.service.concreteness_lexicon import ConcretenessLexicon
from colors_of_meaning.resources import get_resource_path

DEFAULT_CONCRETENESS_RESOURCE = "concreteness_norms.tsv"
NEUTRAL_CONCRETENESS = 3.0
_WORD_PATTERN = re.compile(r"[a-z]+")


def _parse_concreteness_value(value: str) -> Optional[float]:
    try:
        return float(value)
    except ValueError:
        return None


class BrysbaertConcretenessLexicon(ConcretenessLexicon):
    def __init__(self, resource_name: str = DEFAULT_CONCRETENESS_RESOURCE) -> None:
        self._scores = self._load_scores(resource_name)

    def score(self, text: str) -> float:
        known = [self._scores[token] for token in self._tokenize(text) if token in self._scores]
        if not known:
            return NEUTRAL_CONCRETENESS
        return sum(known) / len(known)

    @staticmethod
    def _load_scores(resource_name: str) -> Dict[str, float]:
        path = get_resource_path(resource_name)
        scores: Dict[str, float] = {}
        with open(path, "r", encoding="utf-8") as norms_file:
            for line in norms_file:
                entry = BrysbaertConcretenessLexicon._parse_entry(line)
                if entry is not None:
                    scores[entry[0]] = entry[1]
        return scores

    @staticmethod
    def _parse_entry(line: str) -> Optional[Tuple[str, float]]:
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 2:
            return None
        value = _parse_concreteness_value(fields[1])
        if value is None:
            return None
        return fields[0].lower(), value

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _WORD_PATTERN.findall(text.lower())
