import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine

BASE_DIR = os.path.abspath(Path(__file__).parents[4])
load_dotenv(os.path.join(BASE_DIR, ".env"))
sys.path.append(BASE_DIR)

config = context.config

config.set_main_option("sqlalchemy.url", os.environ["DATABASE_URL"])
if config.config_file_name is None:
    raise ValueError("The config file_name cannot be None")

config_file_name = str(config.config_file_name)
fileConfig(config_file_name)

from src.backend.infrastructure.psql.db import Base
from src.backend.infrastructure.psql.models import *

target_metadata = Base.metadata
db_schema = os.environ.get("DB_SCHEMA", None)


def run_migrations_offline():
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        include_schemas=True,
        version_table_schema=db_schema
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online():
    connectable = AsyncEngine(
        engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
            future=True,
        )
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
