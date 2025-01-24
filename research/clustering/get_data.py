from pathlib import Path

import sqlite3


def get_data_from_regex101():
    sql_query = """
        select 
            r.regex,
            r.dialect,
            r.title,
            r.description
        from regexes r;
        """

    db = sqlite3.connect(Path('..', '..', 'regex101.db'))
    cursor = db.cursor()

    res = cursor.execute(sql_query)
    regex101_regexes = res.fetchall()

    db.commit()
    db.close()

    return regex101_regexes, ('regex', 'dialect', 'title', 'description')
