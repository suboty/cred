from abc import ABC

from domain.repositories import AbstractRepository
from domain.entities.sources.regexlib import *
from infrastructure.psql.models.sources.regexlib import RegexLibModel


class RegexLibRepositoryInterface(
    AbstractRepository[
        RegexLibCreate, RegexLibUpdate, RegexLib, RegexLibFilterSchema, RegexLibModel
    ],
    ABC
):
    ...
