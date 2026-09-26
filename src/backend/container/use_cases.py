from dependency_injector import containers, providers

from use_cases.regexes import (
    CreateRegexesUseCase,
    GetRegexesUseCase,
    UpdateRegexesUseCase,
    DeleteRegexesUseCase,
    BulkCreateRegexesUseCase,
    BulkUpdateRegexesUseCase,
    GetPaginatedRegexesUseCase,
    GetFilteredRegexesUseCase,
)

from use_cases.sources.regex101 import (
    CreateRegex101UseCase,
    GetRegex101UseCase,
    UpdateRegex101UseCase,
    DeleteRegex101UseCase,
    BulkCreateRegex101UseCase,
    BulkUpdateRegex101UseCase,
    GetPaginatedRegex101UseCase,
    GetFilteredRegex101UseCase,
    BulkCreateFromRegex101ParserUseCase
)

from use_cases.sources.regexlib import (
    CreateRegexLibUseCase,
    GetRegexLibUseCase,
    UpdateRegexLibUseCase,
    DeleteRegexLibUseCase,
    BulkCreateRegexLibUseCase,
    BulkUpdateRegexLibUseCase,
    GetPaginatedRegexLibUseCase,
    GetFilteredRegexLibUseCase,
    BulkCreateFromRegexLibParserUseCase
)


class UseCaseContainer(containers.DeclarativeContainer):
    services = providers.DependenciesContainer()
    repositories = providers.DependenciesContainer()

    # regexes
    create_regexes_use_case = providers.Factory(
        CreateRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    get_regexes_use_case = providers.Factory(
        GetRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    update_regexes_use_case = providers.Factory(
        UpdateRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    delete_regexes_use_case = providers.Factory(
        DeleteRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    bulk_create_regexes_use_case = providers.Factory(
        BulkCreateRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    bulk_update_regexes_use_case = providers.Factory(
        BulkUpdateRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    get_paginated_regexes_use_case = providers.Factory(
        GetPaginatedRegexesUseCase,
        regexes_service=services.regexes_service,
    )
    get_filtered_regexes_use_case = providers.Factory(
        GetFilteredRegexesUseCase,
        regexes_service=services.regexes_service,
    )

    # regex101
    create_regex101_use_case = providers.Factory(
        CreateRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    get_regex101_use_case = providers.Factory(
        GetRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    update_regex101_use_case = providers.Factory(
        UpdateRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    delete_regex101_use_case = providers.Factory(
        DeleteRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    bulk_create_regex101_use_case = providers.Factory(
        BulkCreateRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    bulk_update_regex101_use_case = providers.Factory(
        BulkUpdateRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    get_paginated_regex101_use_case = providers.Factory(
        GetPaginatedRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    get_filtered_regex101_use_case = providers.Factory(
        GetFilteredRegex101UseCase,
        regex101_service=services.regex101_service,
    )
    bulk_create_from_regex101_parser_use_case = providers.Factory(
        BulkCreateFromRegex101ParserUseCase,
        regex101_service=services.regex101_service,
    )

    # regexlib
    create_regexlib_use_case = providers.Factory(
        CreateRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    get_regexlib_use_case = providers.Factory(
        GetRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    update_regexlib_use_case = providers.Factory(
        UpdateRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    delete_regexlib_use_case = providers.Factory(
        DeleteRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    bulk_create_regexlib_use_case = providers.Factory(
        BulkCreateRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    bulk_update_regexlib_use_case = providers.Factory(
        BulkUpdateRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    get_paginated_regexlib_use_case = providers.Factory(
        GetPaginatedRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    get_filtered_regexlib_use_case = providers.Factory(
        GetFilteredRegexLibUseCase,
        regexlib_service=services.regexlib_service,
    )
    bulk_create_from_regexlib_parser_use_case = providers.Factory(
        BulkCreateFromRegexLibParserUseCase,
        regexlib_service=services.regexlib_service,
    )
