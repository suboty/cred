from abc import ABC

from src.backend.use_cases import UseCase
from src.backend.domain.services.sources.regexlib import RegexLibService
from src.backend.domain.entities.sources.regexlib import *


class BaseRegexLibUseCase(UseCase, ABC):
    def __init__(self, regexlib_service: RegexLibService):
        self.regexlib_service = regexlib_service


class CreateRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(self, obj: RegexLibCreate) -> RegexLib | None:
        return await self.regexlib_service.create(obj=obj)


class GetRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self, obj_filter: RegexLibFilterSchema | None = None
    ) -> RegexLib | None:
        return await self.regexlib_service.get(obj_filter=obj_filter)


class UpdateRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self,
            obj: RegexLibUpdate,
            obj_filter: RegexLibFilterSchema | None = None,
    ) -> RegexLib | None:
        return await self.regexlib_service.update(
            obj=obj,
            obj_filter=obj_filter,
        )


class DeleteRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self,
            obj_id: int | None = None,
            obj_filter: RegexLibFilterSchema | None = None,
    ) -> None:
        return await self.regexlib_service.delete(
            obj_id=obj_id,
            obj_filter=obj_filter,
        )


class BulkCreateRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(self, objects: list[RegexLibCreate]) -> list[RegexLib | None]:
        return await self.regexlib_service.bulk_create(objects=objects)


class BulkUpdateRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self,
            updates: list[RegexLibUpdate],
            obj_filter: RegexLibFilterSchema | None = None,
    ) -> list[RegexLib | None]:
        return await self.regexlib_service.bulk_update(
            updates=updates,
            obj_filter=obj_filter,
        )


class GetPaginatedRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexLibFilterSchema | None = None,
    ) -> tuple[list[RegexLib] | None, int]:
        return await self.regexlib_service.get_paginated_items(
            page=page,
            size=size,
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
        )


class GetFilteredRegexLibUseCase(BaseRegexLibUseCase):
    async def execute(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexLibFilterSchema | None = None,
            limit: int | None = None,
    ) -> list[RegexLib] | None:
        return await self.regexlib_service.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit,
        )
