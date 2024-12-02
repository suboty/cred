import os
import json
import datetime
from pathlib import Path

from db import DBRepository, EntityMeta, Regexes


if __name__ == '__main__':
    path_to_regexes = Path(
        'regex101',
        'regexes'
    )

    db = DBRepository(
        database_url='sqlite:///regex101.db',
        entity_meta=EntityMeta,
    )

    regexes = os.listdir(path_to_regexes)
    regexes_len = len(regexes)

    for i, regex in enumerate(regexes):

        if i % 100 == 0:
            print(f'Process {i}/{regexes_len}')

        with open(Path(path_to_regexes, regex), 'r') as regex_file:
            data = json.load(regex_file)

            db.create_regex_row(
                meta=Regexes(
                    regex=data['regex'],
                    test_string=data['testString'],
                    flags=data['flags'],
                    delimiter=data['delimiter'],
                    dialect=data['flavor'],
                    title=data['libraryTitle'],
                    description=data['libraryDescription'],
                    created_at=str(datetime.datetime.now()),
                )
            )
