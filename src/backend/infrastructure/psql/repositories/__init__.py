from abc import ABC
from typing import Callable, AsyncContextManager
from contextlib import asynccontextmanager

from sqlalchemy import select, delete, and_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import Select

from logger import logger
from domain.repositories import *


class RepositoryException(Exception):
    ...


class SQLAlchemyRepository(
    AbstractRepository[
        CreateSchema,
        UpdateSchema,
        ReadSchema,
        FilterSchema,
        DataBaseModel
    ],
    ABC
):
    def __init__(
            self,
            session_factory: Callable[[], AsyncContextManager[AsyncSession]],
            model: DataBaseModel,
            entity: ReadSchema
    ):
        super().__init__()
        self.session_factory = session_factory
        self.model = model
        self.entity = entity

    def _convert(self, db_obj: DataBaseModel) -> ReadSchema:
        return self.entity.model_validate(db_obj)

    def _apply_filters(
            self, query: Select, obj_filter: FilterSchema | None
    ) -> Select:
        if not obj_filter:
            return query

        filter_dict = obj_filter.model_dump(exclude_unset=True)
        if filter_dict:
            conditions = []
            for field, value in filter_dict.items():
                if hasattr(self.model, field):
                    conditions.append(
                        getattr(self.model, field) == value
                    )

            if conditions:
                query = query.where(and_(*conditions))

        return query

    @asynccontextmanager
    async def _get_session(self):
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()

    async def create(self, obj: CreateSchema) -> ReadSchema | None:
        async with self._get_session() as db_session:
            try:
                obj_data = obj.model_dump()
                db_obj = self.model(**obj_data)
                db_session.add(db_obj)
                await db_session.commit()
                await db_session.refresh(db_obj)
                return self._convert(db_obj)
            except IntegrityError as e:
                await db_session.rollback()
                logger.error(
                    f"Integrity error while creating object: {e}"
                )
                raise RepositoryException(
                    "Object already exists or violates constraints"
                ) from e
            except Exception as e:
                await db_session.rollback()
                logger.error(f"Error while creating object: {e}")
                raise RepositoryException(
                    f"Failed to create object: {str(e)}"
                ) from e

    async def get(
            self, obj_id: int, obj_filter: FilterSchema | None = None
    ) -> ReadSchema | None:
        async with self._get_session() as db_session:
            try:
                query = select(self.model)
                query = self._apply_filters(query, obj_filter)
                query = query.where(
                    *[getattr(self.model, 'id') == obj_id]
                )
                result = await db_session.execute(query)
                db_obj = result.scalar_one_or_none()
                return self._convert(db_obj) if db_obj else None
            except Exception as e:
                logger.error(f"Error while getting object: {e}")
                raise RepositoryException(
                    f"Failed to get object: {str(e)}"
                ) from e

    async def update(
            self, obj: UpdateSchema, obj_filter: FilterSchema | None = None
    ) -> ReadSchema | None:
        async with self._get_session() as db_session:
            try:
                query = select(self.model)
                query = self._apply_filters(query, obj_filter)
                result = await db_session.execute(query)
                db_obj = result.scalar_one_or_none()

                if db_obj is None:
                    return None

                update_data = obj.model_dump(exclude_unset=True)
                for field, value in update_data.items():
                    if hasattr(db_obj, field):
                        setattr(db_obj, field, value)

                await db_session.commit()
                await db_session.refresh(db_obj)
                return self._convert(db_obj)
            except Exception as e:
                await db_session.rollback()
                logger.error(f"Error while updating object: {e}")
                raise RepositoryException(
                    f"Failed to update object: {str(e)}"
                ) from e

    async def delete(
            self,
            obj_id: int | None = None,
            obj_filter: FilterSchema | None = None
    ) -> None:
        async with self._get_session() as db_session:
            try:
                if obj_id is not None:
                    query = delete(self.model).where(self.model.id == obj_id)
                else:
                    query = delete(self.model)
                    if obj_filter:
                        subquery = select(self.model.id)
                        subquery = self._apply_filters(subquery, obj_filter)
                        query = query.where(self.model.id.in_(subquery))

                result = await db_session.execute(query)
                await db_session.commit()

                if len(result.all()) == 0:
                    logger.warning("No objects found for deletion")

            except Exception as e:
                await db_session.rollback()
                logger.error(f"Error while deleting object: {e}")
                raise RepositoryException(
                    f"Failed to delete object: {str(e)}"
                ) from e

    async def bulk_create(
            self, objects: list[CreateSchema]
    ) -> list[ReadSchema | None]:
        async with self._get_session() as db_session:
            try:
                db_objects = []
                for obj in objects:
                    obj_data = obj.model_dump()
                    db_obj = self.model(**obj_data)
                    db_objects.append(db_obj)

                db_session.add_all(db_objects)
                await db_session.flush()

                for db_obj in db_objects:
                    await db_session.refresh(db_obj)

                await db_session.commit()
                return [self._convert(db_obj) for db_obj in db_objects]

            except IntegrityError as e:
                await db_session.rollback()
                logger.error(f"Integrity error in bulk create: {e}")
                raise RepositoryException(
                    "One or more objects violate constraints"
                ) from e
            except Exception as e:
                await db_session.rollback()
                logger.error(f"Error in bulk create: {e}")
                raise RepositoryException(
                    f"Failed to bulk create objects: {str(e)}"
                ) from e

    async def bulk_update(
            self,
            updates: list[UpdateSchema],
            obj_filter: FilterSchema | None = None
    ) -> list[ReadSchema | None]:
        async with self._get_session() as db_session:
            try:
                if obj_filter:
                    query = select(self.model)
                    query = self._apply_filters(query, obj_filter)
                    result = await db_session.execute(query)
                    db_objects = result.scalars().all()

                    if not db_objects:
                        return []

                    update_data = updates[0].model_dump(
                        exclude_unset=True
                    ) if updates else {}
                    for db_obj in db_objects:
                        for field, value in update_data.items():
                            if hasattr(db_obj, field):
                                setattr(db_obj, field, value)

                    await db_session.commit()
                    return [self._convert(db_obj) for db_obj in db_objects]

                updated_objects = []
                for update_obj in updates:
                    if hasattr(update_obj, 'id') and update_obj.id is not None: # noqa
                        query = select(self.model).where(
                            self.model.id == update_obj.id
                        )
                        result = await db_session.execute(query)
                        db_obj = result.scalar_one_or_none()

                        if db_obj:
                            update_data = update_obj.model_dump(exclude_unset=True)
                            update_data.pop('id', None)
                            for field, value in update_data.items():
                                if hasattr(db_obj, field):
                                    setattr(db_obj, field, value)
                            updated_objects.append(db_obj)

                await db_session.commit()
                return [self._convert(db_obj) for db_obj in updated_objects]

            except Exception as e:
                await db_session.rollback()
                logger.error(f"Error in bulk update: {e}")
                raise RepositoryException(
                    f"Failed to bulk update objects: {str(e)}"
                ) from e

    async def get_paginated_items(
            self,
            page: int,
            size: int,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: FilterSchema | None = None,
    ) -> tuple[list[ReadSchema | None], int]:
        async with self._get_session() as db_session:
            try:
                query = select(self.model)
                query = self._apply_filters(query, obj_filter)

                if sort_field and hasattr(self.model, sort_field):
                    sort_column = getattr(self.model, sort_field)
                    if sort_descending:
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())
                else:
                    query = query.order_by(self.model.id.asc())

                count_query = select(func.count()).select_from(self.model)
                count_query = self._apply_filters(count_query, obj_filter)
                total = (await db_session.execute(count_query)).scalar() or 0

                offset = (page - 1) * size
                query = query.offset(offset).limit(size)
                result = await db_session.execute(query)

                db_objects = result.scalars().all()
                return [self._convert(db_obj) for db_obj in db_objects], total

            except Exception as e:
                logger.error(f"Error in get_paginated_items: {e}")
                raise RepositoryException(
                    f"Failed to get paginated items: {str(e)}"
                ) from e

    async def get_filtered_items(
            self,
            sort_field: str | None = None,
            sort_descending: bool | None = None,
            obj_filter: FilterSchema | None = None,
            limit: int | None = None,
    ) -> list[ReadSchema | None]:
        async with self._get_session() as db_session:
            try:
                query = select(self.model)
                query = self._apply_filters(query, obj_filter)

                if sort_field and hasattr(self.model, sort_field):
                    sort_column = getattr(self.model, sort_field)
                    if sort_descending:
                        query = query.order_by(sort_column.desc())
                    else:
                        query = query.order_by(sort_column.asc())
                else:
                    query = query.order_by(self.model.id.asc())

                if limit is not None and limit > 0:
                    query = query.limit(limit)

                result = await db_session.execute(query)
                db_objects = result.scalars().all()
                return [self._convert(db_obj) for db_obj in db_objects]

            except Exception as e:
                logger.error(f"Error in get_filtered_items: {e}")
                raise RepositoryException(
                    f"Failed to get filtered items: {str(e)}"
                ) from e
