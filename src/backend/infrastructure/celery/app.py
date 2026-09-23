from celery import Celery
from celery import Task

from logger import logger
from infrastructure.celery.config import imports as celery_modules
from infrastructure.celery.config import celery_settings
from container import init_container

container = init_container()


class DBSessionTask(Task):
    def __call__(self, *args, **kwargs):
        try:
            container.wire(modules=celery_modules)
            return self.run(*args, **kwargs)
        except Exception as exc:
            logger.error(f"Task {self.name} failed: {exc}", exc_info=True)
            raise


celery = Celery(
    "service", broker=celery_settings.TASK_BROKER_URL, task_cls=DBSessionTask
)
celery.config_from_object("infrastructure.celery.celery_config")

