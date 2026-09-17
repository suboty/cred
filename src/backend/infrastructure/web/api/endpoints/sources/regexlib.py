import logging
from typing import Any

from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends, HTTPException, status

from use_cases.sources.regexlib import *
from container import Container
from infrastructure.web.schemas.sources.regexlib import *
from domain.entities.sources.regexlib import *
from infrastructure.web.schemas import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    summary="Get regex by filtering fields from regexlib",
    responses={
        status.HTTP_200_OK: {"model": RegexLibResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_regex(
        regex_id: int,
        filter_schema: RegexLibFilterSchema = Depends(),
        use_case: GetRegexLibUseCase = Depends(
            Provide[Container.use_cases.get_regexlib_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(obj_id=regex_id, obj_filter=filter_schema)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex getting from regexlib: {e}",
        ) from e


@router.post(
    "/",
    summary="Create regex from regexlib",
    responses={
        status.HTTP_200_OK: {"model": RegexLibResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def create_regex(
        request_body: RegexLibCreateRequest,
        use_case: CreateRegexLibUseCase = Depends(
            Provide[Container.use_cases.create_regexlib_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=RegexLibCreate.model_validate(request_body)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex creating from regexlib: {e}",
        ) from e


@router.patch(
    "/",
    summary="Update regex from regexlib",
    responses={
        status.HTTP_200_OK: {"model": RegexLibResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def update_regex(
        request_body: RegexLibUpdateRequest,
        filter_schema: RegexLibFilterSchema = Depends(),
        use_case: UpdateRegexLibUseCase = Depends(
            Provide[Container.use_cases.update_regexlib_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=RegexLibUpdate.model_validate(request_body),
            obj_filter=filter_schema,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex updating from regexlib: {e}",
        ) from e


@router.delete(
    "/",
    summary="Delete regex from regexlib",
    responses={
        status.HTTP_200_OK: {"model": SuccessResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def delete_regex(
        regex_id: int,
        filter_schema: RegexLibFilterSchema = Depends(),
        use_case: DeleteRegexLibUseCase = Depends(
            Provide[Container.use_cases.delete_regexlib_use_case]
        ),
) -> Any:
    try:
        await use_case.execute(
            obj_id=regex_id,
            obj_filter=filter_schema,
        )
        return SuccessResponse(
            success=True,
            message=f"Regex with ID <{regex_id}> from regexlib is successfully deleted",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex deleting from regexlib: {e}",
        ) from e


@router.post(
    "/bulk",
    summary="Bulk creating regexes from regexlib",
    responses={
        status.HTTP_200_OK: {"model": list[RegexLibResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_creating_regexes(
        objects: list[RegexLibCreateRequest],
        use_case: BulkCreateRegexLibUseCase = Depends(
            Provide[Container.use_cases.bulk_create_regexlib_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            objects=[
                RegexLibCreate.model_validate(request_body)
                for request_body in objects
            ]
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk creating from regexlib: {e}",
        ) from e


@router.patch(
    "/bulk",
    summary="Bulk updating regexes from regexlib",
    responses={
        status.HTTP_200_OK: {"model": list[RegexLibResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_updating_regexes(
        objects: list[RegexLibUpdateRequest],
        filter_schema: RegexLibFilterSchema = Depends(),
        use_case: BulkUpdateRegexLibUseCase = Depends(
            Provide[Container.use_cases.bulk_update_regexlib_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            updates=[
                RegexLibUpdate.model_validate(request_body)
                for request_body in objects
            ],
            obj_filter=filter_schema,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk updating from regexlib: {e}",
        ) from e


@router.get(
    "/paginated",
    summary="Get paginated list of regexes from regexlib",
    responses={
        status.HTTP_200_OK: {"model": list[RegexLibResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_paginated_regexes(
        page: int = 1,
        size: int = 10,
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: RegexLibFilterSchema = Depends(),
        use_case: GetPaginatedRegexLibUseCase = Depends(
            Provide[Container.use_cases.get_paginated_regexlib_use_case]
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
            detail=f"Error while paginated regex getting from regexlib: {e}",
        ) from e


@router.get(
    "/filtered",
    summary="Get filtered list of regexes from regexlib",
    responses={
        status.HTTP_200_OK: {"model": list[RegexLibResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_filtered_regexes(
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: RegexLibFilterSchema = Depends(),
        limit: int | None = None,
        use_case: GetFilteredRegexLibUseCase = Depends(
            Provide[Container.use_cases.get_filtered_regexlib_use_case]
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
            detail=f"Error while filtered regex getting from regexlib: {e}",
        ) from e
