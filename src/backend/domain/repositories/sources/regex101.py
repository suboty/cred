from abc import ABC

from domain.repositories import AbstractRepository
from domain.entities.sources.regex101 import *
from infrastructure.psql.models.sources.regex101 import Regex101Model


class Regex101RepositoryInterface(
    AbstractRepository[
        Regex101Create, Regex101Update, Regex101, Regex101FilterSchema, Regex101Model
    ],
    ABC
):
    ...
