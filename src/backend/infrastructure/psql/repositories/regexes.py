from src.backend.infrastructure.psql.repositories import SQLAlchemyRepository

from src.backend.domain.entities.regexes import *
from src.backend.domain.repositories.regexes import RegexesRepositoryInterface
from src.backend.infrastructure.psql.models.regexes import RegexesModel


class RegexesSqlAlchemyRepository(
    SQLAlchemyRepository[
        RegexCreate, RegexUpdate, Regex, RegexesFilterSchema, RegexesModel
    ],
    RegexesRepositoryInterface
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=RegexesModel, entity=Regex, **kwargs) # noqa
