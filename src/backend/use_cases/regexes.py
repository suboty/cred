from abc import ABC

from src.backend.use_cases import UseCase
from src.backend.domain.services.regexes import RegexesService
from src.backend.domain.entities.regexes import *


class BaseRegexUseCase(UseCase, ABC):
    def __init__(self, regexes_service: RegexesService):
        self.regexes_service = regexes_service


class CreateRegexesUseCase(BaseRegexUseCase):
    async def execute(self, obj: RegexCreate) -> Regex | None:
        return await self.regexes_service.create(obj=obj)


class GetRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self, obj_filter: RegexesFilterSchema | None = None
    ) -> Regex | None:
        return await self.regexes_service.get(obj_filter=obj_filter)


class UpdateRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self,
            obj: RegexUpdate,
            obj_filter: RegexesFilterSchema | None = None,
    ) -> Regex | None:
        return await self.regexes_service.update(
            obj=obj,
            obj_filter=obj_filter,
        )


class DeleteRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self,
            obj_id: int | None = None,
            obj_filter: RegexesFilterSchema | None = None,
    ) -> None:
        return await self.regexes_service.delete(
            obj_id=obj_id,
            obj_filter=obj_filter,
        )


class BulkCreateRegexesUseCase(BaseRegexUseCase):
    async def execute(self, objects: list[RegexCreate]) -> list[Regex | None]:
        return await self.regexes_service.bulk_create(objects=objects)


class BulkUpdateRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self,
            updates: list[RegexUpdate],
            obj_filter: RegexesFilterSchema | None = None,
    ) -> list[Regex | None]:
        return await self.regexes_service.bulk_update(
            updates=updates,
            obj_filter=obj_filter,
        )


class GetPaginatedRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexesFilterSchema | None = None,
    ) -> tuple[list[Regex] | None, int]:
        return await self.regexes_service.get_paginated_items(
            page=page,
            size=size,
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
        )


class GetFilteredRegexesUseCase(BaseRegexUseCase):
    async def execute(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexesFilterSchema | None = None,
            limit: int | None = None,
    ) -> list[Regex] | None:
        return await self.regexes_service.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit,
        )
