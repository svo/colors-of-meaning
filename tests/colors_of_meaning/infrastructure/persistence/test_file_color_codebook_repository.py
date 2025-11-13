from pathlib import Path

from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import FileColorCodebookRepository
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.lab_color import LabColor


class TestFileColorCodebookRepository:
    def test_should_save_and_load_codebook(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))
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

    def test_should_return_none_when_codebook_not_found(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))

        result = repo.load("nonexistent")

        assert result is None

    def test_should_check_if_codebook_exists(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        assert not repo.exists("test_codebook")
        repo.save(codebook, "test_codebook")
        assert repo.exists("test_codebook")

    def test_should_delete_codebook(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)
        repo.save(codebook, "test_codebook")

        repo.delete("test_codebook")

        assert not repo.exists("test_codebook")

    def test_should_handle_delete_when_codebook_does_not_exist(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))

        repo.delete("nonexistent")

        assert True

    def test_should_create_base_path_if_not_exists(self, tmp_path: Path) -> None:
        base_path = tmp_path / "nested" / "path"
        FileColorCodebookRepository(base_path=str(base_path))

        assert base_path.exists()

    def test_should_add_pkl_extension_if_missing(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        repo.save(codebook, "test_codebook")

        assert (tmp_path / "test_codebook.pkl").exists()

    def test_should_not_add_extension_if_already_present(self, tmp_path: Path) -> None:
        repo = FileColorCodebookRepository(base_path=str(tmp_path))
        colors = [LabColor(l=0.0, a=0.0, b=0.0)]
        codebook = ColorCodebook(colors=colors, num_bins=1)

        repo.save(codebook, "test_codebook.pkl")

        assert (tmp_path / "test_codebook.pkl").exists()
        assert not (tmp_path / "test_codebook.pkl.pkl").exists()
