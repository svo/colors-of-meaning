from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationSample:
    text: str
    label: int
    split: str

    def __post_init__(self) -> None:
        if not self.text:
            raise ValueError("text cannot be empty")
        if self.label < 0:
            raise ValueError(f"label must be non-negative, got {self.label}")
        if self.split not in ("train", "test", "validation"):
            raise ValueError(f"split must be 'train', 'test', or 'validation', got {self.split}")
