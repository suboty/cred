import logging

from pydantic_settings import SettingsConfigDict, BaseSettings


class LoggerSettings(BaseSettings):
    LOGGING_LEVEL: str = "WARNING"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    def get_logging_level(self) -> int:
        return getattr(logging, self.LOGGING_LEVEL.upper(), logging.WARNING)


logger_settings = LoggerSettings()

logging.basicConfig(
    level=logger_settings.get_logging_level(),
    format="%(asctime)s : %(levelname)s : %(module)s : %(funcName)s: %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)

logger = logging.getLogger(__name__)
