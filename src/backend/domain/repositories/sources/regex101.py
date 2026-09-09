from abc import ABC

from src.backend.domain.repositories import AbstractRepository
from src.backend.domain.entities.sources.regex101 import *
from src.backend.infrastructure.psql.models.sources.regex101 import Regex101Model


class Regex101RepositoryInterface(
    AbstractRepository[
        Regex101Create, Regex101Update, Regex101, Regex101FilterSchema, Regex101Model
    ],
    ABC
):
    ...
