from infrastructure.psql.repositories import SQLAlchemyRepository

from domain.entities.sources.regexlib import *
from domain.repositories.sources.regexlib import RegexLibRepositoryInterface
from infrastructure.psql.models.sources.regexlib import RegexLibModel


class RegexLibSqlAlchemyRepository(
    SQLAlchemyRepository[
        RegexLibCreate, RegexLibUpdate, RegexLib, RegexLibFilterSchema, RegexLibModel
    ],
    RegexLibRepositoryInterface
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=RegexLibModel, entity=Regex101, **kwargs) # noqa
