from domain.repositories.sources.regex101 import Regex101RepositoryInterface
from domain.entities.sources.regex101 import *


class Regex101Service:
    def __init__(
            self,
            regex101_repository: Regex101RepositoryInterface
    ):
        self.regex101_repository = regex101_repository

    async def create(self, obj: Regex101Create) -> Regex101 | None:
        return await self.regex101_repository.create(obj=obj)

    async def get(
            self,
            obj_id: int,
            obj_filter: Regex101FilterSchema | None
    ) -> Regex101 | None:
        return await self.regex101_repository.get(
            obj_id=obj_id,
            obj_filter=obj_filter
        )

    async def update(
            self, obj: Regex101Update, obj_filter: Regex101FilterSchema | None
    ) -> Regex101 | None:
        return await self.regex101_repository.update(obj=obj, obj_filter=obj_filter)

    async def delete(self, obj_id: int | None, obj_filter: Regex101FilterSchema | None) -> None:
        return await self.regex101_repository.delete(obj_id=obj_id, obj_filter=obj_filter)

    async def bulk_create(self, objects: list[Regex101Create]) -> list[Regex101 | None]:
        return await self.regex101_repository.bulk_create(objects=objects)

    async def bulk_update(
            self,
            updates: list[Regex101Update],
            obj_filter: Regex101FilterSchema | None
    ) -> list[Regex101 | None]:
        return await self.regex101_repository.bulk_update(updates=updates, obj_filter=obj_filter)

    async def get_paginated_items(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: Regex101FilterSchema | None = None,
    ) -> tuple[list[Regex101] | None, int]:
        return await self.regex101_repository.get_paginated_items(
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
            obj_filter: Regex101FilterSchema | None = None,
            limit: int | None = None,
    ) -> list[Regex101] | None:
        return await self.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit
        )
