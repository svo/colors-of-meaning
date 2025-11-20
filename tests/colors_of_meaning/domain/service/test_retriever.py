import pytest
from colors_of_meaning.domain.service.retriever import Retriever
from colors_of_meaning.domain.model.evaluation_sample import EvaluationSample


class ConcreteRetriever(Retriever):
    """Concrete implementation for testing the abstract Retriever class."""

    def fit(self, samples: list[EvaluationSample]) -> None:
        super().fit(samples)

    def search(self, query: EvaluationSample, k: int) -> list[tuple[EvaluationSample, float]]:
        super().search(query, k)
        return []


class TestRetriever:
    def test_should_raise_not_implemented_error_for_fit(self) -> None:
        retriever = ConcreteRetriever()
        samples = [EvaluationSample(text="test", label=0, split="train")]

        with pytest.raises(NotImplementedError):
            retriever.fit(samples)

    def test_should_raise_not_implemented_error_for_search(self) -> None:
        retriever = ConcreteRetriever()
        query = EvaluationSample(text="query", label=0, split="test")

        with pytest.raises(NotImplementedError):
            retriever.search(query, k=5)
