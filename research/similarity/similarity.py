import os
import argparse
from pathlib import Path
from typing import List, Callable, Dict, Optional

import numpy as np

from logger import logger
from get_data import get_data_from_database
from generator import Generator
from preprocessing.sre import SreParser
from preprocessing.custom import CustomTranslator
from metrics.string import StringSimilarity
from metrics.graph import GraphSimilarity
from db import (
    EntityMeta,
    ResearchRepository,
    Regexes,
    Results,
)


queries = {
    'all': 0,
    'similar': 1,
    'same_construction': 2,
    'same_construction_percentage': 3
}


DATA_LIMIT = 1000
IS_NEED_REGEX_SAVING = True


def get_similarity_matrix(
        _ids: List,
        _regexes: List,
        similarity_func: Callable,
        _db: ResearchRepository,
        regex_type: str,
        metric_name: str,
        kwargs_func: Optional[Dict] = None,
):
    if kwargs_func is None:
        kwargs_func = {}
    mem = []
    _i = 0
    for i_x, x in enumerate(_regexes):
        mem.append(i_x)
        for i_y, y in enumerate(_regexes):
            if i_x == i_y or i_y in mem:
                continue
            _i += 1
            if _i % 10 == 0:
                logger.info(f'--- Processed <{_i}> pairs')
            _result = round(
                similarity_func(x, y, **kwargs_func),
                2
            )
            db.create_result(
                meta=Results(
                    regex1=_ids[i_x],
                    regex2=_ids[i_y],
                    regex_type=regex_type,
                    metric_name=metric_name,
                    metric_value=_result
                )
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cred-clustering')

    # regex source
    parser.add_argument('--regexSource', type=str, default='regex101')

    # regex group (query name)
    parser.add_argument('--regexGroup', type=str, default='all')

    # regex construction for same_construction grouping
    parser.add_argument('--regexConstruction', type=str, default=None)

    # regex construction thresholds for same_construction_percentage grouping
    parser.add_argument('--regexConstructionThresholdUpper', type=str, default=None)
    parser.add_argument('--regexConstructionThresholdLower', type=str, default=None)

    # init objects
    args = parser.parse_args()

    kwargs = {}
    if args.regexConstruction:
        kwargs.setdefault('kwargs_construction', args.regexConstruction)
    if args.regexConstructionThresholdUpper and args.regexConstructionThresholdLower:
        kwargs.setdefault('__THRESHOLD_UPPER__', args.regexConstructionThresholdUpper)
        kwargs.setdefault('__THRESHOLD_LOWER__', args.regexConstructionThresholdLower)

    # get data
    data = get_data_from_database(
        database=args.regexSource,
        query_index=queries.get(args.regexGroup),
        **kwargs
    )

    if len(data) > DATA_LIMIT:
        indexes = np.random.randint(0, len(data) - 1, DATA_LIMIT)
        new_data = []
        for index in indexes:
            new_data.append(data[index])
        data = new_data

    logger.info(f'Work with <{len(data)}> samples')

    # init objects for similarity measuring
    os.makedirs('tmp', exist_ok=True)

    sp = SreParser()
    ct = CustomTranslator()
    g = Generator()
    db = ResearchRepository(
        database_url=f'sqlite:///tmp/similarity.db',
        entity_meta=EntityMeta,
    )

    # get string regexes
    regexes = [x[0] for x in data]
    equal_regexes = g(regexes, True)

    # parsing errors may occur when obtaining regular expression graphs.

    for i_key, key in enumerate(equal_regexes.keys()):
        str_regexes = equal_regexes[key]

        original = db.create_regex(
            meta=Regexes(regex=key)
        )

        ids = [original['id']]
        for x in str_regexes:
            ids.append(db.create_regex(
                meta=Regexes(
                    regex=x,
                    prototype=original['id']
                )
            )['id'])

        # get graph regexes by SRE parser
        new_str_regexes = []
        sre_regexes = []
        errors = 0
        for i, x in enumerate(str_regexes):
            try:
                sre_regex = sp(x)
            except Exception as e:
                logger.warning(
                    f'Wrong regular expression parsing with SRE Parser.'
                    f'\n\tRegex: <{x}>.'
                    f'\n\tError: <{e}>.'
                    f'\n\t# of error: <{errors}>'
                )
                errors += 1
                continue
            sre_regexes.append(sre_regex)
            new_str_regexes.append(str_regexes[i])
        str_regexes = new_str_regexes

        # get graph regexes by Custom translator
        new_str_regexes = []
        new_sre_regexes = []
        translator_regexes = []
        errors = 0
        for i, x in enumerate(str_regexes):
            try:
                translator_regex = ct(x)
            except Exception as e:
                logger.warning(
                    f'Wrong regular expression parsing with Custom Translator.'
                    f'\n\tRegex: <{x}>.'
                    f'\n\tError: <{e}>.'
                    f'\n\t# of error: <{errors}>'
                )
                errors += 1
                continue
            translator_regexes.append(translator_regex)
            new_str_regexes.append(str_regexes[i])
            new_sre_regexes.append(sre_regexes[i])
        str_regexes = new_str_regexes
        sre_regexes = new_sre_regexes

        if len(str_regexes) in [0, 1]:
            logger.warning('Skip. Not enough regexes')
            continue
        else:
            logger.info(f'After parsing work with <{len(str_regexes)}> samples')

        # Regex as a string
        str_regexes = [key] + str_regexes

        # Levenshtein
        logger.info(f'<{i_key}> Levenshtein. Work with <{len(str_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=str_regexes,
            similarity_func=StringSimilarity.get_distance,
            _db=db,
            _ids=ids,
            metric_name='levenshtein',
            regex_type='str'
        )

        # Hamming
        logger.info(f'<{i_key}> Hamming. Work with <{len(str_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=str_regexes,
            similarity_func=StringSimilarity.get_hamming_distance,
            _db=db,
            _ids=ids,
            metric_name='hamming',
            regex_type='str'
        )

        # Jaro
        logger.info(f'<{i_key}> Jaro. Work with <{len(str_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=str_regexes,
            similarity_func=StringSimilarity.get_jaro_similarity,
            _db=db,
            _ids=ids,
            metric_name='jaro',
            regex_type='str'
        )

        # Jaro-Winkler
        logger.info(f'<{i_key}> Jaro-Winkler. Work with <{len(str_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=str_regexes,
            similarity_func=StringSimilarity.get_jaro_similarity,
            _db=db,
            _ids=ids,
            metric_name='jaro_winkler',
            regex_type='str',
            kwargs_func={'is_jaro_winkler': True}
        )

        # Regex as a graph

        # GED for SRE Parser
        logger.info(f'<{i_key}> GED for SRE Parser. Work with <{len(sre_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=sre_regexes,
            similarity_func=GraphSimilarity.get_graph_edit_distance,
            _db=db,
            _ids=ids,
            metric_name='sre_parser',
            regex_type='ast',
            kwargs_func={'is_optimize': True}
        )

        # GED for Custom Translator
        logger.info(f'<{i_key}> GED for Custom Translator. Work with <{len(translator_regexes)}> samples.')
        get_similarity_matrix(
            _regexes=translator_regexes,
            _db=db,
            _ids=ids,
            metric_name='custom_translator',
            regex_type='ast',
            similarity_func=GraphSimilarity.get_graph_edit_distance,
            kwargs_func={'is_optimize': True}
        )
