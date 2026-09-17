from dependency_injector import containers, providers

from infrastructure.psql.repositories.regexes import (
    RegexesSqlAlchemyRepository
)
from infrastructure.psql.repositories.sources.regex101 import (
    Regex101SqlAlchemyRepository
)
from infrastructure.psql.repositories.sources.regexlib import (
    RegexLibSqlAlchemyRepository
)

class RepositoryContainer(containers.DeclarativeContainer):
    db: providers.DependenciesContainer = providers.DependenciesContainer()

    regex_repository = providers.Factory(
        RegexesSqlAlchemyRepository,
        session_factory=db.psql_db_client.provided.session,
    )

    # sources

    regex101_repository = providers.Factory(
        Regex101SqlAlchemyRepository,
        session_factory=db.psql_db_client.provided.session,
    )

    regexlib_repository = providers.Factory(
        RegexLibSqlAlchemyRepository,
        session_factory=db.psql_db_client.provided.session,
    )
