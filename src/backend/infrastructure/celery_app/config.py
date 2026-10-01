from pathlib import Path

from kombu import Queue
from datetime import timedelta
from celery.schedules import crontab
from pydantic_settings import SettingsConfigDict, BaseSettings


class CelerySettings(BaseSettings):
    TASK_BROKER_URL: str = "amqp://guest:guest@localhost:5672/"
    TASK_QUEUE_NAME: str = "periodic_queue"

    CRONTAB_REGEX101_PARSING: str = "sun"
    CRONTAB_REGEXLIB_PARSING: str = "sat"

    model_config = SettingsConfigDict(
        env_file=Path("src", "backend", ".env"),
        env_file_encoding="utf-8", extra="allow"
    )


celery_settings = CelerySettings()  # type: ignore

imports = (
    "infrastructure.celery_app.tasks.parse_regex101",
    "infrastructure.celery_app.tasks.parse_regexlib",
)

accept_content = ["json", "msgpack", "yaml"]
task_serializer = "json"
result_serializer = "json"
enable_utc = True
timezone = "Europe/Moscow"

broker_pool_limit = 1
worker_prefetch_multiplier = 1
broker_heartbeat = None
broker_connection_timeout = 40
broker_connection_retry_on_startup = True

_UNIT_TO_KWARG = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
}


def build_schedule(value: str | int):
    if isinstance(value, int):
        return timedelta(seconds=value)

    value = (value or "").strip()
    if not value:
        raise ValueError("Empty schedule value")

    if isinstance(value, str):
        if value.isdigit():
            return timedelta(seconds=int(value))

        if len(value) >= 2 and value[-1].lower() in _UNIT_TO_KWARG:
            number = value[:-1]
            if number.isdigit():
                return timedelta(**{_UNIT_TO_KWARG[value[-1].lower()]: int(number)})

    return crontab(minute="0", hour="0", day_of_week=value)


beat_schedule: dict = {
    "regex101_parsing_task": {
        "task": "infrastructure.celery_app.tasks.parse_regex101.regex101_parsing_task",
        "schedule": build_schedule(celery_settings.CRONTAB_REGEX101_PARSING),
        "options": {"queue": celery_settings.TASK_QUEUE_NAME},
    },
    "regexlib_parsing_task": {
        "task": "infrastructure.celery_app.tasks.parse_regexlib.regexlib_parsing_task",
        "schedule": build_schedule(celery_settings.CRONTAB_REGEXLIB_PARSING),
        "options": {"queue": celery_settings.TASK_QUEUE_NAME},
    },
}

task_queues = [
    Queue(
        celery_settings.TASK_QUEUE_NAME,
        routing_key=celery_settings.TASK_QUEUE_NAME,
    ),
]

task_routes = {
    "infrastructure.celery_app.tasks.*": {
        "queue": celery_settings.TASK_QUEUE_NAME
    },
}
