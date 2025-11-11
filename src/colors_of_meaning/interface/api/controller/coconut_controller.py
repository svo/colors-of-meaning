from typing import Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.security import HTTPBasicCredentials
from pydantic import UUID4

from lagom import Container

from colors_of_meaning.application.use_case.coconut_use_case import CreateCoconutUseCase, GetCoconutUseCase
from colors_of_meaning.interface.api.data_transfer_object.coconut_data_transfer_object import (
    CoconutApiRequestDataTransferObject,
    CoconutApiResponseDataTransferObject,
)


class CoconutController:
    def __init__(
        self,
        get_coconut_use_case: GetCoconutUseCase,
        create_coconut_use_case: CreateCoconutUseCase,
        authentication_dependency: Optional[Callable[[Optional[HTTPBasicCredentials]], None]] = None,
    ) -> None:
        self.get_coconut_use_case = get_coconut_use_case
        self.create_coconut_use_case = create_coconut_use_case
        self.authentication_dependency = authentication_dependency
        self.router = APIRouter(prefix="/coconut", tags=["coconut"])
        self._register_routes()

    def _register_routes(self) -> None:
        dependencies = [Depends(self.authentication_dependency)] if self.authentication_dependency else []

        self.router.add_api_route(
            "/{id}",
            self.get_coconut,
            methods=["GET"],
            response_model=CoconutApiResponseDataTransferObject,
            dependencies=dependencies,
        )

        self.router.add_api_route(
            "/",
            self.create_coconut,
            methods=["POST"],
            status_code=status.HTTP_201_CREATED,
            response_class=Response,
            dependencies=dependencies,
        )

    async def get_coconut(self, id: UUID4) -> CoconutApiResponseDataTransferObject:
        try:
            coconut = self.get_coconut_use_case.execute(id)
            return CoconutApiResponseDataTransferObject.from_domain_model(coconut)
        except Exception as e:
            if "not found" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Coconut with id {id} not found",
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error retrieving coconut: {str(e)}",
            )

    async def create_coconut(self, data_transfer_object: CoconutApiRequestDataTransferObject) -> Response:
        try:
            created_id = self.create_coconut_use_case.execute(data_transfer_object.id)

            response = Response(status_code=status.HTTP_201_CREATED)
            response.headers["Location"] = f"/coconut/{created_id}"

            return response
        except Exception as e:
            if "already exists" in str(e).lower():
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Coconut with id {data_transfer_object.id} already exists",
                )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Error creating coconut: {str(e)}",
            )


def create_coconut_controller(
    container: Container, authentication_dependency: Optional[Callable[[Optional[HTTPBasicCredentials]], None]] = None
) -> CoconutController:
    get_coconut_use_case = container[GetCoconutUseCase]
    create_coconut_use_case = container[CreateCoconutUseCase]

    return CoconutController(
        get_coconut_use_case=get_coconut_use_case,
        create_coconut_use_case=create_coconut_use_case,
        authentication_dependency=authentication_dependency,
    )
