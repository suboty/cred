from infrastructure.psql.repositories import SQLAlchemyRepository

from domain.entities.sources.regex101 import *
from domain.repositories.sources.regex101 import Regex101RepositoryInterface
from infrastructure.psql.models.sources.regex101 import Regex101Model


class Regex101SqlAlchemyRepository(
    SQLAlchemyRepository[
        Regex101Create, Regex101Update, Regex101, Regex101FilterSchema, Regex101Model
    ],
    Regex101RepositoryInterface
):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, model=RegexesModel, entity=Regex101, **kwargs) # noqa
