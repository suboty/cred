import asyncio

from dependency_injector.wiring import inject, Provide

from infrastructure.celery.app import celery
from container import Container


@inject
async def async_regex101_parsing_task(
        get_filtered_regexlib_use_case=Provide[
            Container.use_cases.get_filtered_regexlib_use_case
        ],
):
    await get_filtered_regexlib_use_case.execute()


@celery.task(bind=True)
def regex101_parsing_task(self):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_regex101_parsing_task())
