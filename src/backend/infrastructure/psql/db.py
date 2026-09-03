from asyncio import current_task
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    async_sessionmaker,
    create_async_engine,
    AsyncSession,
    async_scoped_session,
)
from sqlalchemy.orm import declarative_base, declared_attr


class Base:
    id: Any
    __name__: str

    @declared_attr
    def __tablename__(cls) -> str: # noqa
        return cls.__name__.lower()


metadata = MetaData()
Base = declarative_base(cls=Base, metadata=metadata)


class Database:
    def __init__(
        self,
        db_url: str,
        pool_size: int,
        max_overflow: int,
        pool_timeout: int,
        pool_recycle: int,
    ) -> None:
        """
        Initializes a connection to a database.
        :param db_url: URL for the database connection.
        :param pool_size: Number of connections kept open in the pool.
        :param max_overflow: Size of additional connections,
                             that can be opened during high workload.
        :param pool_timeout: Time to wait for a connection to become available to open.
        :param pool_recycle: Time after which a connection is recycled.
        """
        self.db_url = db_url
        self._async_engine = create_async_engine(
            self.db_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
        )
        self._session_factory = async_scoped_session(
            async_sessionmaker(
                self._async_engine,
                autocommit=False,
                autoflush=False,
                expire_on_commit=False,
                class_=AsyncSession,
            ),
            scopefunc=current_task,
        )

    def get_session(self) -> AsyncSession:
        return self._session_factory()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        session: AsyncSession = self._session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
