from abc import ABC

from src.backend.domain.repositories import AbstractRepository
from src.backend.domain.entities.sources.regexlib import *
from src.backend.infrastructure.psql.models.sources.regexlib import RegexLibModel


class RegexLibRepositoryInterface(
    AbstractRepository[
        RegexLibCreate, RegexLibUpdate, RegexLib, RegexLibFilterSchema, RegexLibModel
    ],
    ABC
):
    ...
