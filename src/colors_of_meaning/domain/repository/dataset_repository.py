from abc import ABC, abstractmethod
from typing import List, Optional

from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class DatasetRepository(ABC):
    @abstractmethod
    def get_samples(self, split: str, max_samples: Optional[int] = None) -> List[EvaluationSample]:
        raise NotImplementedError

    @abstractmethod
    def get_label_names(self) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def get_num_classes(self) -> int:
        raise NotImplementedError
