from abc import ABC, abstractmethod
from typing import Tuple, TypeVar, Generic

from pydantic import BaseModel


__all__ = [
    'CreateSchema', 'UpdateSchema', 'ReadSchema', 'FilterSchema',
    'AbstractRepository'
]


CreateSchema = TypeVar('CreateSchema', bound=BaseModel)
UpdateSchema = TypeVar('UpdateSchema', bound=BaseModel)
ReadSchema = TypeVar('ReadSchema', bound=BaseModel)
FilterSchema = TypeVar('FilterSchema', bound=BaseModel)


class AbstractRepository(
    ABC,
    Generic[CreateSchema, UpdateSchema, ReadSchema, FilterSchema]
):
    @abstractmethod
    async def create(self, obj: CreateSchema) -> ReadSchema:
        pass

    @abstractmethod
    async def get(self, obj_filter: FilterSchema | None) -> ReadSchema | None:
        pass

    @abstractmethod
    async def update(
            self, obj: UpdateSchema, obj_filter: FilterSchema | None
    ) -> ReadSchema | None:
        pass

    @abstractmethod
    async def delete(self, obj_id: int | None, obj_filter: FilterSchema | None) -> None:
        pass

    @abstractmethod
    async def bulk_create(self, objects: list[CreateSchema]) -> list[ReadSchema]:
        pass

    @abstractmethod
    async def bulk_update(self, updates: list[UpdateSchema]) -> list[ReadSchema]:
        pass

    @abstractmethod
    async def get_paginated_items(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: FilterSchema | None = None,
    ) -> Tuple[list[ReadSchema], int]:
        pass

    @abstractmethod
    async def get_filtered_items(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: FilterSchema | None = None,
            limit: int | None = None,
    ) -> list[ReadSchema]:
        pass
