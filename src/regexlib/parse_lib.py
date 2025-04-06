import time
import logging
import datetime
from pathlib import Path

import zeep

from db import DBRepository, EntityMeta, Regexes


if __name__ == '__main__':

    logging.getLogger('zeep').setLevel(logging.ERROR)

    path_to_regexes = Path(
        'regexlib',
        'regexes'
    )

    db = DBRepository(
        database_url='sqlite:///regexlib.db',
        entity_meta=EntityMeta,
    )

    wsdl = 'https://regexlib.com/WebServices.asmx?wsdl'
    client = zeep.Client(wsdl=wsdl)
    number_of_regexes = 3000
    result_number = 0

    print(f'-- Run regexlib parsing')
    t0 = time.time()
    try:
        result = client.service.ListAllAsXml(number_of_regexes)
    except Exception as e:
        print(f'Error while parsing!')
        exit(1)
    print(f'-- regexlib parsing is finished, elapsed time: {time.time()-t0}')

    for i, regex in enumerate(result):

        if i % 100 == 0:
            print(f'-- Process {i}/{number_of_regexes}')

        result_number += 1

        _ = db.create_regex_row(
            meta=Regexes(
                old_id=regex['Id'],
                title=regex['Title'],
                pattern=regex['Pattern'],
                matching_text=regex['MatchingText'],
                non_matching_text=regex['NonMatchingText'],
                description=regex['Description'],
                is_dirty=regex['IsDirty'],
                author_name=regex['AuthorName'],
                rating=regex['Rating'],
                date_modified=regex['DateModified'],
                created_at=str(datetime.datetime.now()),
            )
        )

    print(f'-- Result number of regexes: {result_number}')
