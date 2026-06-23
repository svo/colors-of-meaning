import logging
import pickle  # nosec B403
import sys
import uuid
from typing import List, Optional, cast

import uvicorn
from fastapi import APIRouter, FastAPI
from lagom import Container

from colors_of_meaning.application.use_case.compare_documents_use_case import CompareDocumentsUseCase
from colors_of_meaning.application.use_case.health_use_case import HealthUseCase
from colors_of_meaning.application.use_case.query_by_palette_use_case import QueryByPaletteUseCase
from colors_of_meaning.domain.health.health_checker import HealthChecker
from colors_of_meaning.domain.model.color_codebook import ColorCodebook
from colors_of_meaning.domain.model.colored_document import ColoredDocument
from colors_of_meaning.domain.service.distance_calculator import DistanceCalculator
from colors_of_meaning.infrastructure.ml.jensen_shannon_distance_calculator import (
    JensenShannonDistanceCalculator,
)
from colors_of_meaning.infrastructure.ml.wasserstein_distance_calculator import (
    WassersteinDistanceCalculator,
)
from colors_of_meaning.infrastructure.persistence.file_color_codebook_repository import (
    FileColorCodebookRepository,
)
from colors_of_meaning.infrastructure.security.basic_authentication import (
    BasicAuthenticator,
    SecurityDependency,
    get_basic_authenticator,
)
from colors_of_meaning.infrastructure.system.health_factory import create_health_checker
from colors_of_meaning.interface.api.controller.health_controller import create_health_controller
from colors_of_meaning.interface.api.controller.query_controller import (
    create_query_controller,
    create_unavailable_query_controller,
)
from colors_of_meaning.shared.configuration import get_application_setting_provider

logger = logging.getLogger(__name__)

APPLICATION_TITLE = "Colors of Meaning API"
APPLICATION_VERSION = "1.0.0"
CODEBOOK_NAME = "codebook_4096"
CORPUS_ARTIFACT_PATH = "artifacts/encoded/test_documents.pkl"
DISTANCE_METRIC = "wasserstein"
FALLBACK_BINS_PER_DIMENSION = 16
SINKHORN_REGULARISATION: Optional[float] = None
SMOOTHING_EPSILON = 1e-8
CORPUS_UNAVAILABLE_DETAIL = "Color retrieval is unavailable: encoded corpus artifact is missing"


def _correlation_id() -> str:
    return str(uuid.uuid4())


def _load_query_codebook() -> ColorCodebook:
    codebook = FileColorCodebookRepository().load(CODEBOOK_NAME)
    if codebook is not None:
        return codebook

    logger.warning(
        "Codebook artifact absent; falling back to uniform grid",
        extra={"correlation_id": _correlation_id(), "expected_codebook": CODEBOOK_NAME},
    )
    return ColorCodebook.create_uniform_grid(FALLBACK_BINS_PER_DIMENSION)


def _load_corpus() -> Optional[List[ColoredDocument]]:
    try:
        with open(CORPUS_ARTIFACT_PATH, "rb") as artifact:
            return cast(List[ColoredDocument], pickle.load(artifact))  # nosec B301 nosemgrep
    except FileNotFoundError:
        logger.warning(
            "Encoded corpus artifact absent; query endpoint degraded",
            extra={"correlation_id": _correlation_id(), "expected_corpus": CORPUS_ARTIFACT_PATH},
        )
        return None


def _select_distance_calculator(metric: str, codebook: ColorCodebook) -> DistanceCalculator:
    if metric != "wasserstein":
        return JensenShannonDistanceCalculator(smoothing_epsilon=SMOOTHING_EPSILON)
    return WassersteinDistanceCalculator(codebook=codebook, sinkhorn_reg=SINKHORN_REGULARISATION)


def get_container() -> Container:
    container = Container()

    codebook = _load_query_codebook()

    container[ColorCodebook] = lambda: codebook
    container[DistanceCalculator] = lambda: _select_distance_calculator(DISTANCE_METRIC, codebook)  # type: ignore
    container[CompareDocumentsUseCase] = CompareDocumentsUseCase
    container[QueryByPaletteUseCase] = QueryByPaletteUseCase

    authenticator = get_basic_authenticator()
    security_dependency = SecurityDependency(authenticator)
    container[BasicAuthenticator] = lambda: authenticator
    container[SecurityDependency] = lambda: security_dependency

    health_checker = create_health_checker()
    container[HealthChecker] = lambda: health_checker  # type: ignore
    container[HealthUseCase] = HealthUseCase

    return container


global_container = get_container()


def get_global_container() -> Container:
    return global_container


def _build_query_router(container: Container, corpus: Optional[List[ColoredDocument]]) -> APIRouter:
    if corpus is None:
        return create_unavailable_query_controller(CORPUS_UNAVAILABLE_DETAIL)
    return create_query_controller(container[QueryByPaletteUseCase], corpus)


def _log_query_readiness(codebook: ColorCodebook, corpus: Optional[List[ColoredDocument]]) -> None:
    logger.info(
        "Query API wiring complete",
        extra={
            "correlation_id": _correlation_id(),
            "codebook_name": CODEBOOK_NAME,
            "codebook_num_bins": codebook.num_bins,
            "corpus_documents": len(corpus) if corpus is not None else "absent",
            "distance_metric": DISTANCE_METRIC,
        },
    )


def create_app() -> FastAPI:
    application = FastAPI(title=APPLICATION_TITLE, version=APPLICATION_VERSION)

    container = get_container()
    codebook = container[ColorCodebook]
    corpus = _load_corpus()

    application.include_router(_build_query_router(container, corpus))
    application.include_router(create_health_controller(container[HealthUseCase]))

    _log_query_readiness(codebook, corpus)

    return application


app = create_app()


def main(args: list) -> None:
    settings_provider = get_application_setting_provider()
    reload_setting = settings_provider.get("reload")
    host_setting = settings_provider.get("host")

    uvicorn.run(
        "colors_of_meaning.interface.api.main:app",
        reload=reload_setting,
        host=host_setting,
    )


def run() -> None:
    main(sys.argv[1:])
