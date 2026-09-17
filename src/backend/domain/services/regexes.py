from domain.repositories.regexes import RegexesRepositoryInterface
from domain.entities.regexes import *


class RegexesService:
    def __init__(
            self,
            regexes_repository: RegexesRepositoryInterface
    ):
        self.regexes_repository = regexes_repository

    async def create(self, obj: RegexCreate) -> Regex | None:
        return await self.regexes_repository.create(obj=obj)

    async def get(self, obj_filter: RegexesFilterSchema | None) -> Regex | None:
        return await self.regexes_repository.get(obj_filter=obj_filter)

    async def update(
            self, obj: RegexUpdate, obj_filter: RegexesFilterSchema | None
    ) -> Regex | None:
        return await self.regexes_repository.update(obj=obj, obj_filter=obj_filter)

    async def delete(self, obj_id: int | None, obj_filter: RegexesFilterSchema | None) -> None:
        return await self.regexes_repository.delete(obj_id=obj_id, obj_filter=obj_filter)

    async def bulk_create(self, objects: list[RegexCreate]) -> list[Regex | None]:
        return await self.regexes_repository.bulk_create(objects=objects)

    async def bulk_update(
            self,
            updates: list[RegexUpdate],
            obj_filter: RegexesFilterSchema | None
    ) -> list[Regex | None]:
        return await self.regexes_repository.bulk_update(updates=updates, obj_filter=obj_filter)

    async def get_paginated_items(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: RegexesFilterSchema | None = None,
    ) -> tuple[list[Regex] | None, int]:
        return await self.regexes_repository.get_paginated_items(
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
            obj_filter: RegexesFilterSchema | None = None,
            limit: int | None = None,
    ) -> list[Regex] | None:
        return await self.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit
        )
