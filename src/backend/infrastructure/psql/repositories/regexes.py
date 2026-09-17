from infrastructure.psql.repositories import SQLAlchemyRepository

from domain.entities.regexes import *
from domain.repositories.regexes import RegexesRepositoryInterface
from infrastructure.psql.models.regexes import RegexesModel


class RegexesSqlAlchemyRepository(
    SQLAlchemyRepository[
        RegexCreate, RegexUpdate, Regex, RegexesFilterSchema, RegexesModel
    ],
    RegexesRepositoryInterface
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=RegexesModel, entity=Regex, **kwargs) # noqa
