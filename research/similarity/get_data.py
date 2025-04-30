from pathlib import Path

import sqlite3


database_queries = [
    # get all regexes
    """select r.__REGEX_COLUMN__ 
    from regexes r 
    where r.__REGEX_COLUMN__ != ' ' 
    and r.__REGEX_COLUMN__ != '' 
    and r.__REGEX_COLUMN__ is not null;""",

    # get completely similar regexes
    """select r.__REGEX_COLUMN__, count(*) 
    from regexes r 
    group by r.__REGEX_COLUMN__ 
    order by 2 desc;""",

    # get similar regexes with same construction
    """select r.__REGEX_COLUMN__, count(*) 
    from (select lower(__REGEX_COLUMN__) as __REGEX_COLUMN__ from regexes ) as r 
    where r.__REGEX_COLUMN__ like '%\w%'
    group by r.__REGEX_COLUMN__
    order by 2 desc;""",
]


def get_data_from_database(
        query_index: int,
        database: str,
):
    sql_query = database_queries[query_index]
    match database:
        case 'regexlib':
            sql_query = sql_query.replace('__REGEX_COLUMN__', 'pattern')
        case 'regex101':
            sql_query = sql_query.replace('__REGEX_COLUMN__', 'regex')
        case _:
            raise NotImplementedError

    db = sqlite3.connect(Path('..', '..', f'{database}.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regexes = res.fetchall()

    db.commit()
    db.close()

    return regexes



