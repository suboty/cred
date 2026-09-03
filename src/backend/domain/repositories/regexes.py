from abc import ABC

from src.backend.domain.repositories import AbstractRepository
from src.backend.domain.entities.regexes import *


class RegexLibRepositoryInterface(
    AbstractRepository[
        RegexCreate, RegexUpdate, Regex, RegexesFilterSchema
    ],
    ABC
):
    ...
