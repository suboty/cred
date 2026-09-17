import logging
from typing import Any

from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter, Depends, HTTPException, status

from use_cases.regexes import *
from container import Container
from infrastructure.web.schemas.regexes import *
from domain.entities.regexes import *
from infrastructure.web.schemas import ErrorResponse, SuccessResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get(
    "/",
    summary="Get regex by ID and filtering fields from CRED",
    responses={
        status.HTTP_200_OK: {"model": RegexResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_regex(
        regex_id: int,
        filter_schema: RegexesFilterSchema = Depends(),
        use_case: GetRegexesUseCase = Depends(
            Provide[Container.use_cases.get_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(obj_id=regex_id, obj_filter=filter_schema)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex getting: {e}",
        ) from e


@router.post(
    "/",
    summary="Create regex for CRED",
    responses={
        status.HTTP_200_OK: {"model": RegexResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def create_regex(
        request_body: RegexCreateRequest,
        use_case: CreateRegexesUseCase = Depends(
            Provide[Container.use_cases.create_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=RegexCreate.model_validate(request_body)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex creating: {e}",
        ) from e


@router.patch(
    "/",
    summary="Update regex in CRED",
    responses={
        status.HTTP_200_OK: {"model": RegexResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def update_regex(
        request_body: RegexUpdateRequest,
        filter_schema: RegexesFilterSchema = Depends(),
        use_case: UpdateRegexesUseCase = Depends(
            Provide[Container.use_cases.update_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            obj=RegexUpdate.model_validate(request_body),
            obj_filter=filter_schema
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex updating: {e}",
        ) from e


@router.delete(
    "/",
    summary="Delete regex in CRED",
    responses={
        status.HTTP_200_OK: {"model": SuccessResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def delete_regex(
        regex_id: int,
        filter_schema: RegexesFilterSchema = Depends(),
        use_case: DeleteRegexesUseCase = Depends(
            Provide[Container.use_cases.delete_regexes_use_case]
        ),
) -> Any:
    try:
        await use_case.execute(
            obj_id=regex_id,
            obj_filter=filter_schema
        )
        return SuccessResponse(
            success=True,
            message=f'Regex with ID <{regex_id}> is successfully deleted'
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex deleting: {e}",
        ) from e


@router.post(
    "/bulk",
    summary="Bulk creating regexes for CRED",
    responses={
        status.HTTP_200_OK: {"model": list[RegexResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_creating_regexes(
        objects: list[RegexCreateRequest],
        use_case: BulkCreateRegexesUseCase = Depends(
            Provide[Container.use_cases.bulk_create_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(objects=[
            RegexCreate.model_validate(request_body)
            for request_body in objects
        ])
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk creating: {e}",
        ) from e


@router.patch(
    "/bulk",
    summary="Bulk updating regexes for CRED",
    responses={
        status.HTTP_200_OK: {"model": list[RegexResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def bulk_updating_regexes(
        objects: list[RegexUpdateRequest],
        filter_schema: RegexesFilterSchema = Depends(),
        use_case: BulkUpdateRegexesUseCase = Depends(
            Provide[Container.use_cases.bulk_create_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            updates=[
                RegexUpdate.model_validate(request_body)
                for request_body in objects
            ],
            obj_filter=filter_schema
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while regex bulk creating: {e}",
        ) from e


@router.get(
    "/paginated",
    summary="Get paginated list of regexes from CRED",
    responses={
        status.HTTP_200_OK: {"model": list[RegexResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_paginated_regexes(
        page: int = 1,
        size: int = 10,
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: RegexesFilterSchema = Depends(),
        use_case: GetPaginatedRegexesUseCase = Depends(
            Provide[Container.use_cases.get_paginated_regexes_use_case]
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
            detail=f"Error while paginated regex getting: {e}",
        ) from e


@router.get(
    "/filtered",
    summary="Get filtered list of regexes from CRED",
    responses={
        status.HTTP_200_OK: {"model": list[RegexResponse]},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
@inject
async def get_filtered_regexes(
        sort_field: str | None = None,
        sort_descending: bool | None = None,
        obj_filter: RegexesFilterSchema = Depends(),
        limit: int | None = None,
        use_case: GetFilteredRegexesUseCase = Depends(
            Provide[Container.use_cases.get_filtered_regexes_use_case]
        ),
) -> Any:
    try:
        return await use_case.execute(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error while filtered regex getting: {e}",
        ) from e
