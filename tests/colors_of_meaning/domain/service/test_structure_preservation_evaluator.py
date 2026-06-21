import pytest

from colors_of_meaning.domain.service.structure_preservation_evaluator import (
    StructurePreservationEvaluator,
)


class TestStructurePreservationEvaluator:
    def test_should_be_abstract(self) -> None:
        with pytest.raises(TypeError):
            StructurePreservationEvaluator()  # type: ignore
