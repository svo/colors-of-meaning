from colors_of_meaning.infrastructure.persistence.in_memory.in_memory_color_codebook_repository import (
    InMemoryColorCodebookRepository,
)
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor


class TestInMemoryColorCodebookRepository:
    def test_should_save_and_load_codebook(self) -> None:
        repo = InMemoryColorCodebookRepository()
        colors = [
            LabColor(l=0.0, a=0.0, b=0.0),
            LabColor(l=50.0, a=0.0, b=0.0),
        ]
        codebook = ColorCodebook(colors=colors, num_bins=2)

        repo.save(codebook, "test_codebook")
        loaded = repo.load("test_codebook")

        assert loaded is not None
        assert loaded.num_bins == 2
        assert len(loaded.colors) == 2

    def test_should_return_none_when_codebook_not_found(self) -> None:
        repo = InMemoryColorCodebookRepository()

        result = repo.load("nonexistent")

        assert result is None

    def test_should_check_if_codebook_exists(self) -> None:
        repo = InMemoryColorCodebookRepository()
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        assert not repo.exists("test_codebook")
        repo.save(codebook, "test_codebook")
        assert repo.exists("test_codebook")

    def test_should_delete_codebook(self) -> None:
        repo = InMemoryColorCodebookRepository()
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)
        repo.save(codebook, "test_codebook")

        repo.delete("test_codebook")

        assert not repo.exists("test_codebook")

    def test_should_handle_delete_when_codebook_does_not_exist(self) -> None:
        repo = InMemoryColorCodebookRepository()

        repo.delete("nonexistent")

        assert True

    def test_should_clear_all_codebooks(self) -> None:
        repo = InMemoryColorCodebookRepository()
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook1 = ColorCodebook(colors=colors, num_bins=1)
        codebook2 = ColorCodebook(colors=colors, num_bins=1)
        repo.save(codebook1, "codebook1")
        repo.save(codebook2, "codebook2")

        repo.clear()

        assert not repo.exists("codebook1")
        assert not repo.exists("codebook2")

    def test_should_overwrite_existing_codebook(self) -> None:
        repo = InMemoryColorCodebookRepository()
        colors1 = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook1 = ColorCodebook(colors=colors1, num_bins=1)
        colors2 = [LabColor(l=50.0, a=0.0, b=0.0), LabColor(l=100.0, a=0.0, b=0.0)]
        codebook2 = ColorCodebook(colors=colors2, num_bins=2)

        repo.save(codebook1, "test")
        repo.save(codebook2, "test")
        loaded = repo.load("test")

        assert loaded is not None
        assert loaded.num_bins == 2
