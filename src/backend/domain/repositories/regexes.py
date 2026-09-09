from abc import ABC

from src.backend.domain.repositories import AbstractRepository
from src.backend.domain.entities.regexes import *
from src.backend.infrastructure.psql.models.regexes import RegexesModel


class RegexesRepositoryInterface(
    AbstractRepository[
        RegexCreate, RegexUpdate, Regex, RegexesFilterSchema, RegexesModel
    ],
    ABC
):
    ...
