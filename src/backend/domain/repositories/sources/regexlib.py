from abc import ABC

from src.backend.domain.repositories import AbstractRepository
from src.backend.domain.entities.sources.regexlib import *


class RegexLibRepositoryInterface(
    AbstractRepository[
        RegexLibCreate, RegexLibUpdate, RegexLib, RegexLibFilterSchema
    ],
    ABC
):
    ...
