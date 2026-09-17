from abc import ABC

from domain.repositories import AbstractRepository
from domain.entities.regexes import *
from infrastructure.psql.models.regexes import RegexesModel


class RegexesRepositoryInterface(
    AbstractRepository[
        RegexCreate, RegexUpdate, Regex, RegexesFilterSchema, RegexesModel
    ],
    ABC
):
    ...
