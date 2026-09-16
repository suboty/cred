from src.backend.domain.repositories.sources.regexlib import RegexLibRepositoryInterface
from src.backend.domain.entities.sources.regexlib import *


class RegexLibService:
    def __init__(
            self,
            regexlib_repository: RegexLibRepositoryInterface
    ):
        self.regexlib_repository = regexlib_repository

    async def create(self, obj: RegexLibCreate) -> RegexLib | None:
        return await self.regexlib_repository.create(obj=obj)

    async def get(self, obj_filter: RegexLibFilterSchema | None) -> RegexLib | None:
        return await self.regexlib_repository.get(obj_filter=obj_filter)

    async def update(
            self, obj: RegexLibUpdate, obj_filter: RegexLibFilterSchema | None
    ) -> RegexLib | None:
        return await self.regexlib_repository.update(obj=obj, obj_filter=obj_filter)

    async def delete(self, obj_id: int | None, obj_filter: RegexLibFilterSchema | None) -> None:
        return await self.regexlib_repository.delete(obj_id=obj_id, obj_filter=obj_filter)

    async def bulk_create(self, objects: list[RegexLibCreate]) -> list[RegexLib | None]:
        return await self.regexlib_repository.bulk_create(objects=objects)

    async def bulk_update(
            self,
            updates: list[RegexLibUpdate],
            obj_filter: RegexLibFilterSchema | None
    ) -> list[RegexLib | None]:
        return await self.regexlib_repository.bulk_update(updates=updates, obj_filter=obj_filter)

    async def get_paginated_items(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexLibFilterSchema | None = None,
    ) -> tuple[list[RegexLib] | None, int]:
        return await self.regexlib_repository.get_paginated_items(
            page=page,
            size=size,
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter
        )

    async def get_filtered_items(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexLibFilterSchema | None = None,
            limit: int | None = None,
    ) -> list[RegexLib] | None:
        return await self.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit
        )
