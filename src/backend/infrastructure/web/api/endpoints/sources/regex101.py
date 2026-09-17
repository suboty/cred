import logging
from typing import Any

from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends, HTTPException, status

from use_cases.sources.regex101 import *
from container import Container
from infrastructure.web.schemas.sources.regex101 import *
from domain.entities.sources.regex101 import *
from infrastructure.web.schemas import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    summary="Get regex by filtering fields from regex101",
    responses={
        status.HTTP_200_OK: {"model": Regex101Response},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_regex(
        filter_schema: Regex101FilterSchema | None = None,
        use_case: GetRegex101UseCase = Depends(
            Provide[Container.use_cases.get_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(obj_filter=filter_schema)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex getting from regex101: {e}",
        ) from e


@router.post(
    "/",
    summary="Create regex from regex101",
    responses={
        status.HTTP_200_OK: {"model": Regex101Response},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def create_regex(
        request_body: Regex101CreateRequest,
        use_case: CreateRegex101UseCase = Depends(
            Provide[Container.use_cases.create_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=Regex101Create.model_validate(request_body)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex creating from regex101: {e}",
        ) from e


@router.patch(
    "/",
    summary="Update regex from regex101",
    responses={
        status.HTTP_200_OK: {"model": Regex101Response},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def update_regex(
        request_body: Regex101UpdateRequest,
        filter_schema: Regex101FilterSchema | None = None,
        use_case: UpdateRegex101UseCase = Depends(
            Provide[Container.use_cases.update_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=Regex101Update.model_validate(request_body),
            obj_filter=filter_schema,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex updating from regex101: {e}",
        ) from e


@router.delete(
    "/",
    summary="Delete regex from regex101",
    responses={
        status.HTTP_200_OK: {"model": SuccessResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def delete_regex(
        regex_id: int,
        filter_schema: Regex101FilterSchema | None = None,
        use_case: DeleteRegex101UseCase = Depends(
            Provide[Container.use_cases.delete_regex101_use_case]
        ),
) -> Any:
    try:
        await use_case.execute(
            obj_id=regex_id,
            obj_filter=filter_schema,
        )
        return SuccessResponse(
            success=True,
            message=f"Regex with ID <{regex_id}> from regex101 is successfully deleted",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex deleting from regex101: {e}",
        ) from e


@router.post(
    "/bulk",
    summary="Bulk creating regexes from regex101",
    responses={
        status.HTTP_200_OK: {"model": list[Regex101Response]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_creating_regexes(
        objects: list[Regex101CreateRequest],
        use_case: BulkCreateRegex101UseCase = Depends(
            Provide[Container.use_cases.bulk_create_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            objects=[
                Regex101Create.model_validate(request_body)
                for request_body in objects
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk creating from regex101: {e}",
        ) from e


@router.patch(
    "/bulk",
    summary="Bulk updating regexes from regex101",
    responses={
        status.HTTP_200_OK: {"model": list[Regex101Response]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_updating_regexes(
        objects: list[Regex101UpdateRequest],
        filter_schema: Regex101FilterSchema | None = None,
        use_case: BulkUpdateRegex101UseCase = Depends(
            Provide[Container.use_cases.bulk_update_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            updates=[
                Regex101Update.model_validate(request_body)
                for request_body in objects
            ],
            obj_filter=filter_schema,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk updating from regex101: {e}",
        ) from e


@router.get(
    "/paginated",
    summary="Get paginated list of regexes from regex101",
    responses={
        status.HTTP_200_OK: {"model": list[Regex101Response]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_paginated_regexes(
        page: int = 1,
        size: int = 10,
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: Regex101FilterSchema | None = None,
        use_case: GetPaginatedRegex101UseCase = Depends(
            Provide[Container.use_cases.get_paginated_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            page=page,
            size=size,
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while paginated regex getting from regex101: {e}",
        ) from e


@router.get(
    "/filtered",
    summary="Get filtered list of regexes from regex101",
    responses={
        status.HTTP_200_OK: {"model": list[Regex101Response]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_filtered_regexes(
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: Regex101FilterSchema | None = None,
        limit: int | None = None,
        use_case: GetFilteredRegex101UseCase = Depends(
            Provide[Container.use_cases.get_filtered_regex101_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while filtered regex getting from regex101: {e}",
        ) from e
