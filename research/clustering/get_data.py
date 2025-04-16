import datetime
from pathlib import Path
from typing import Optional

import sqlite3

from db import Regexes


def add_rows_by_filter_word(
    sql_query: str,
    filter_word: Optional[str] = None,
):
    if filter_word:
        if '|' in filter_word:
            filter_words = [x.lower() for x in filter_word.split('|')]
            sql_query += f"where " \
                         f"lower(r.description) like '%{filter_words[0]}%' " \
                         f"or lower(r.title) like '%{filter_words[0]}%'"
            for word in filter_words[1:]:
                sql_query += f" or " \
                             f"lower(r.description) like '%{word}%' " \
                             f"or lower(r.title) like '%{word}%'"
            sql_query += ";"
        else:
            sql_query += f"where lower(r.description) like '%{filter_word}%' or lower(r.title) like '%{filter_word}%';"
    else:
        sql_query += ';'
    return sql_query


def get_data_from_regex101(
        filter_word: Optional[str] = None
):
    sql_query = """
    select 
        r.regex,
        r.dialect,
        r.title,
        r.description
    from regexes r
    """

    sql_query = add_rows_by_filter_word(
        filter_word=filter_word,
        sql_query=sql_query
    )

    db = sqlite3.connect(Path('..', '..', 'regex101.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regex101_regexes = res.fetchall()

    db.commit()
    db.close()

    return regex101_regexes, ('regex', 'dialect', 'title', 'description')


def get_data_from_regexlib(
        filter_word: Optional[str] = None
):
    sql_query = """
    select 
        r.pattern,
        r.rating,
        r.title,
        r.description
    from regexes r
    """

    sql_query = add_rows_by_filter_word(
        filter_word=filter_word,
        sql_query=sql_query
    )

    db = sqlite3.connect(Path('..', '..', 'regexlib.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regexlib_regexes = res.fetchall()

    db.commit()
    db.close()

    return regexlib_regexes, ('pattern', 'rating', 'title', 'description')


def data_to_db(db, regexes, labels, **kwargs):
    for i, regex in enumerate(regexes):
        db.create_regex(Regexes(
            regex=regex,
            label=labels[i],
            is_ast=kwargs.get('is_ast'),
            is_preprocessed=kwargs.get('is_preprocessed'),
            created_at=str(datetime.datetime.now()),
        ))
