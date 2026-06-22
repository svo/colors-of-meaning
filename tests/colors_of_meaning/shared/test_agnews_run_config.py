from pathlib import Path

from assertpy import assert_that

from colors_of_meaning.shared.synesthetic_config import SynestheticConfig

AGNEWS_RUN_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "agnews_run.yaml"


def _load_agnews_run_config() -> SynestheticConfig:
    return SynestheticConfig.from_yaml(str(AGNEWS_RUN_CONFIG_PATH))


class TestAGNewsRunConfig:
    def test_should_pin_seed_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.training.seed).is_equal_to(42)

    def test_should_pin_cpu_device_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.training.device).is_equal_to("cpu")

    def test_should_pin_max_samples_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.dataset.max_samples).is_equal_to(400)

    def test_should_target_ag_news_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.dataset.name).is_equal_to("ag_news")

    def test_should_pin_num_classes_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.supervised_mapper.num_classes).is_equal_to(4)

    def test_should_use_exact_earth_mover_distance_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.distance.sinkhorn_reg).is_none()

    def test_should_pin_codebook_resolution_when_loading_agnews_run_config(self) -> None:
        config = _load_agnews_run_config()

        assert_that(config.codebook.bins_per_dimension).is_equal_to(16)
