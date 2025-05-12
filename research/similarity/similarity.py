import os
import argparse
from pathlib import Path
from typing import List, Callable, Dict, Optional
from itertools import combinations_with_replacement

import numpy as np

from logger import logger
from get_data import get_data_from_database
from generator import Generator
from preprocessing.sre import SreParser
from preprocessing.custom import CustomTranslator
from metrics.string import StringSimilarity
from metrics.graph import GraphSimilarity


queries = {
    'all': 0,
    'similar': 1,
    'same_construction': 2,
    'same_construction_percentage': 3
}


DATA_LIMIT = 50


def get_similarity_matrix(
        name: str,
        regexes: List,
        similarity_func: Callable,
        kwargs_func: Optional[Dict] = None,
):
    if kwargs_func is None:
        kwargs_func = {}
    result = []
    regex_pairs = combinations_with_replacement(regexes, 2)
    regex_pairs = [(x, y) for x, y in regex_pairs if x != y]
    _i = 0
    for x, y in regex_pairs:
        _i += 1
        if _i % 10 == 0:
            logger.info(f'--- Processed <{_i}> samples')
        result.append(similarity_func(x, y, **kwargs_func))
    result = [round(x, 2) for x in result]
    os.makedirs('results', exist_ok=True)
    with open(Path('results', f'{name}_results.similarity'), 'w') as res_file:
        result = [str(x)+'\n' for x in result]
        res_file.writelines(result)
    return result


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
    sp = SreParser()
    ct = CustomTranslator()
    g = Generator()

    # get string regexes
    regexes = [x[0] for x in data]
    equal_regexes = g(regexes, True)

    # parsing errors may occur when obtaining regular expression graphs.

    for key in equal_regexes.keys():
        str_regexes = equal_regexes[key]

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
            os.makedirs('results', exist_ok=True)
            with open(Path('results', f'{key}_regexes.samples'), 'w') as file:
                file.writelines([x + '\n' for x in str_regexes])

            logger.info(f'After parsing work with <{len(str_regexes)}> samples')

        # Regex as a string

        # Levenshtein
        logger.info(f'<{key}> Levenshtein. Work with <{len(str_regexes)}> samples.')
        l_res = get_similarity_matrix(
            name=f'{key}_levenshtein',
            regexes=str_regexes,
            similarity_func=StringSimilarity.get_distance,
        )

        # Hamming
        logger.info(f'<{key}> Hamming. Work with <{len(str_regexes)}> samples.')
        h_res = get_similarity_matrix(
            name=f'{key}_hamming',
            regexes=str_regexes,
            similarity_func=StringSimilarity.get_hamming_distance,
        )

        # Jaro
        logger.info(f'<{key}> Jaro. Work with <{len(str_regexes)}> samples.')
        j_res = get_similarity_matrix(
            name=f'{key}_jaro',
            regexes=str_regexes,
            similarity_func=StringSimilarity.get_jaro_similarity,
        )

        # Jaro-Winkler
        logger.info(f'<{key}> Jaro-Winkler. Work with <{len(str_regexes)}> samples.')
        jw_res = get_similarity_matrix(
            name=f'{key}_jaro_winkler',
            regexes=str_regexes,
            similarity_func=StringSimilarity.get_jaro_similarity,
            kwargs_func={'is_jaro_winkler': True}
        )

        # Regex as a graph

        # GED for SRE Parser
        logger.info(f'<{key}> GED for SRE Parser. Work with <{len(sre_regexes)}> samples.')
        sre_res = get_similarity_matrix(
            name=f'{key}_sre',
            regexes=sre_regexes,
            similarity_func=GraphSimilarity.get_graph_edit_distance,
            kwargs_func={'is_optimize': True}
        )

        # GED for Custom Translator
        logger.info(f'<{key}> GED for Custom Translator. Work with <{len(translator_regexes)}> samples.')
        translator_res = get_similarity_matrix(
            name=f'{key}_custom',
            regexes=translator_regexes,
            similarity_func=GraphSimilarity.get_graph_edit_distance,
            kwargs_func={'is_optimize': True}
        )
