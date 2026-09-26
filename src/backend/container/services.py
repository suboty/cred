from dependency_injector import containers, providers

from domain.services.regexes import RegexesService
from domain.services.sources.regexlib import RegexLibService
from domain.services.sources.regex101 import Regex101Service
from domain.services.parsing.regexlib import RegexLibParser
from domain.services.parsing.regex101 import Regex101Parser


class ServiceContainer(containers.DeclarativeContainer):
    repositories = providers.DependenciesContainer()

    regexes_service = providers.Factory(
        RegexesService,
        regexes_repository=repositories.regex_repository,
    )

    # sources (tables for storage)

    regex101_service = providers.Factory(
        Regex101Service,
        regex101_repository=repositories.regex101_repository,
    )

    regexlib_service = providers.Factory(
        RegexLibService,
        regexlib_repository=repositories.regexlib_repository,
    )

    # source parsing

    regex101_parsing_service = providers.Factory(
        Regex101Parser,
        # TODO: add system variables
    )

    regexlib_parsing_service = providers.Factory(
        RegexLibParser,
        # TODO: add system variables
    )
