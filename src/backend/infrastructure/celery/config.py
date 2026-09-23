from celery.schedules import crontab
from kombu import Queue
from pydantic_settings import SettingsConfigDict, BaseSettings


class CelerySettings(BaseSettings):
    TASK_BROKER_URL: str = "amqp://guest:guest@localhost:5672/"
    TASK_QUEUE_NAME: str = "periodic_queue"

    CRONTAB_REGEX101_PARSING: str = "sun"
    CRONTAB_REGEXLIB_PARSING: str = "sat"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )


celery_settings = CelerySettings()  # type: ignore

imports = (
    "infrastructure.celery.tasks.parse_regex101",
    "infrastructure.celery.tasks.parse_regelib",
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

beat_schedule: dict = {
    "regex101_parsing_task": {
        "task": "infrastructure.celery.tasks.parse_regex101.regex101_parsing_task",
        "schedule": crontab(
            minute="0",
            hour="0",
            day_of_week=celery_settings.CRONTAB_REGEX101_PARSING,
        ),
        "options": {"queue": celery_settings.TASK_QUEUE_NAME},
    },
    "regexlib_parsing_task": {
        "task": "infrastructure.celery.tasks.parse_regexlib.regexlib_parsing_task",
        "schedule": crontab(
            minute="0",
            hour="0",
            day_of_week=celery_settings.CRONTAB_REGEXLIB_PARSING,
        ),
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
    "infrastructure.celery.tasks.parse_regex101.*": {
        "queue": celery_settings.TASK_QUEUE_NAME
    },
    "infrastructure.celery.tasks.parse_regexlib.*": {
        "queue": celery_settings.TASK_QUEUE_NAME
    },
}

