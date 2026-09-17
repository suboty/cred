from dependency_injector import containers, providers

from src.backend.container.databases import DatabaseContainer
from src.backend.container.repositories import RepositoryContainer
from src.backend.container.services import ServiceContainer
from src.backend.container.use_cases import UseCaseContainer


class Container(containers.DeclarativeContainer):
    db = providers.Container(DatabaseContainer)
    repositories = providers.Container(RepositoryContainer, db=db)
    services = providers.Container(ServiceContainer, repositories=repositories)
    use_cases = providers.Container(UseCaseContainer, services=services, repositories=repositories)


def init_container():
    container = Container()
    container.init_resources()
    return container
