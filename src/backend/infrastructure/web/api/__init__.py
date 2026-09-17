from fastapi import APIRouter

from infrastructure.web.api.endpoints import regexes
from infrastructure.web.api.endpoints.sources import regex101
from infrastructure.web.api.endpoints.sources import regexlib


api_router = APIRouter()


api_router.include_router(
    regexes.router, prefix="/regexes", tags=["regexes"]
)
api_router.include_router(
    regex101.router, prefix="/sources/regex101", tags=["/sources/regex101"]
)
api_router.include_router(
    regexlib.router, prefix="/sources/regexlib", tags=["/sources/regexlib"]
)
