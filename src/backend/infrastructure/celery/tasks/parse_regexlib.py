import asyncio

from dependency_injector.wiring import inject, Provide

from infrastructure.celery.app import celery
from container import Container


@inject
async def async_regexlib_parsing_task(
        bulk_create_from_regexlib_parser_use_case=Provide[
            Container.use_cases.bulk_create_from_regexlib_parser_use_case
        ],
):
    await bulk_create_from_regexlib_parser_use_case.execute()


@celery.task(bind=True)
def regexlib_parsing_task(self):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(async_regexlib_parsing_task())
