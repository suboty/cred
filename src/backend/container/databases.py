from dependency_injector import containers, providers
from src.backend.infrastructure.config import (
    db_settings,
)
from src.backend.infrastructure.psql.db import Database


class DatabaseContainer(containers.DeclarativeContainer):
    psql_db_client = providers.Singleton(
        Database,
        db_url=db_settings.DATABASE_URL,
        pool_size=db_settings.DB_POOL_SIZE,
        max_overflow=db_settings.DB_MAX_OVERFLOW,
        pool_timeout=db_settings.DB_POOL_TIMEOUT_SEC,
        pool_recycle=db_settings.DB_POOL_RECYCLE_SEC,
    )
