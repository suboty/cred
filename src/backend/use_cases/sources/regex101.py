from abc import ABC

from use_cases import UseCase
from domain.services.sources.regex101 import Regex101Service
from domain.services.parsing.regex101 import Regex101Parser
from domain.entities.sources.regex101 import *


class BaseRegex101UseCase(UseCase, ABC):
    def __init__(self, regex101_service: Regex101Service):
        self.regex101_service = regex101_service


class CreateRegex101UseCase(BaseRegex101UseCase):
    async def execute(self, obj: Regex101Create) -> Regex101 | None:
        return await self.regex101_service.create(obj=obj)


class GetRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            obj_id: int,
            obj_filter: Regex101FilterSchema | None = None
    ) -> Regex101 | None:
        return await self.regex101_service.get(
            obj_id=obj_id,
            obj_filter=obj_filter
        )


class UpdateRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            obj: Regex101Update,
            obj_filter: Regex101FilterSchema | None = None,
    ) -> Regex101 | None:
        return await self.regex101_service.update(
            obj=obj,
            obj_filter=obj_filter,
        )


class DeleteRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            obj_id: int | None = None,
            obj_filter: Regex101FilterSchema | None = None,
    ) -> None:
        return await self.regex101_service.delete(
            obj_id=obj_id,
            obj_filter=obj_filter,
        )


class BulkCreateRegex101UseCase(BaseRegex101UseCase):
    async def execute(self, objects: list[Regex101Create]) -> list[Regex101 | None]:
        return await self.regex101_service.bulk_create(objects=objects)


class BulkUpdateRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            updates: list[Regex101Update],
            obj_filter: Regex101FilterSchema | None = None,
    ) -> list[Regex101 | None]:
        return await self.regex101_service.bulk_update(
            updates=updates,
            obj_filter=obj_filter,
        )


class GetPaginatedRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: Regex101FilterSchema | None = None,
    ) -> tuple[list[Regex101] | None, int]:
        return await self.regex101_service.get_paginated_items(
            page=page,
            size=size,
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
        )


class GetFilteredRegex101UseCase(BaseRegex101UseCase):
    async def execute(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: Regex101FilterSchema | None = None,
            limit: int | None = None,
    ) -> list[Regex101] | None:
        return await self.regex101_service.get_filtered_items(
            sort_field=sort_field,
            sort_descending=sort_descending,
            obj_filter=obj_filter,
            limit=limit,
        )


class BulkCreateFromRegex101ParserUseCase(BaseRegex101UseCase):
    async def execute(
            self,
            parser_service: Regex101Parser,
            limit_pages: int
    ) -> Regex101ParsingResult:
        parsed_regexes = await parser_service.parse()
        created_regexes = await self.regex101_service.bulk_create(
            objects=[Regex101Create(**x) for x in parsed_regexes]
        )
        return Regex101ParsingResult(
            parsed_regexes=len(parsed_regexes),
            created_regexes=len(created_regexes)
        )
