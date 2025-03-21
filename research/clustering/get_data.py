from pathlib import Path
from typing import Optional

import sqlite3


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

    db = sqlite3.connect(Path('..', '..', 'regex101.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regex101_regexes = res.fetchall()

    db.commit()
    db.close()

    return regex101_regexes, ('regex', 'dialect', 'title', 'description')
