from pathlib import Path

import sqlite3


database_queries = [
    # get all regexes
    """select r.__REGEX_COLUMN__ 
    from regexes r 
    where r.__REGEX_COLUMN__ not in ('', ' ')
    and r.__REGEX_COLUMN__ is not null;""",

    # get completely similar regexes
    """select r.__REGEX_COLUMN__, count(*) 
    from regexes r 
    group by r.__REGEX_COLUMN__ 
    order by 2 desc;""",

    # get similar regexes with same construction
    """select r.__REGEX_COLUMN__, count(*) 
    from (select lower(__REGEX_COLUMN__) as __REGEX_COLUMN__ from regexes ) as r 
    where r.__REGEX_COLUMN__ like '%kwargs_construction%'
    group by r.__REGEX_COLUMN__
    order by 2 desc;""",

    # get similar regexes with same constructions
    # with construction percentage filter
    """with vars as (select 'kwargs_construction' as pattern)
    select
        r.__REGEX_COLUMN__,
        r.pattern_percentage,
        count(*) as count
    from (
        select
            lower(regexes.__REGEX_COLUMN__) as __REGEX_COLUMN__,
            round(cast(length(vars.pattern) as real)/length(regexes.__REGEX_COLUMN__),2) as pattern_percentage
        from regexes, vars
    ) as r, vars
    where
        r.__REGEX_COLUMN__ like format("%s%s%s", '%', vars.pattern, '%')
        and r.pattern_percentage >= __THRESHOLD__
    group by r.__REGEX_COLUMN__
    order by 2 desc, 3 desc;"""
]


def get_data_from_database(
        query_index: int,
        database: str,
        **kwargs
):
    sql_query = database_queries[query_index]
    match database:
        case 'regexlib':
            sql_query = sql_query.replace('__REGEX_COLUMN__', 'pattern')
        case 'regex101':
            sql_query = sql_query.replace('__REGEX_COLUMN__', 'regex')
        case _:
            raise NotImplementedError

    for k, v in kwargs.items():
        sql_query = sql_query.replace(k, v)

    db = sqlite3.connect(Path('..', '..', f'{database}.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regexes = res.fetchall()

    db.commit()
    db.close()

    return regexes
