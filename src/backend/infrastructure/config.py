from pathlib import Path

from pydantic_settings import SettingsConfigDict, BaseSettings


class DbSettings(BaseSettings):
    DATABASE_URL: str
    DB_SCHEMA: str = "public"
    DB_POOL_SIZE: int = 5
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT_SEC: int = 30
    DB_POOL_RECYCLE_SEC: int = 3600

    model_config = SettingsConfigDict(
        env_file=Path("src", "backend", ".env"),
        env_file_encoding="utf-8", extra="allow"
    )


db_settings = DbSettings()  # type: ignore


class AppSettings(BaseSettings):
    PROJECT_NAME: str = "CRED-API"
    PROJECT_VERSION: str = "0.1.0"

    model_config = SettingsConfigDict(
        env_file=Path("src", "backend", ".env"),
        env_file_encoding="utf-8", extra="allow"
    )


app_settings = AppSettings()
