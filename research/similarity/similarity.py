import os
import argparse
import traceback
from pathlib import Path
from typing import List, Callable, Dict, Optional
from itertools import combinations_with_replacement

from tqdm import tqdm

from logger import logger
from get_data import get_data_from_database
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
    for x, y in tqdm(regex_pairs):
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

    # regex construction threshold for same_construction_percentage grouping
    parser.add_argument('--regexConstructionThreshold', type=str, default=None)

    # init objects
    args = parser.parse_args()

    kwargs = {}
    if args.regexConstruction:
        kwargs.setdefault('kwargs_construction', args.regexConstruction)
    if args.regexConstructionThreshold:
        kwargs.setdefault('__THRESHOLD__', args.regexConstructionThreshold)

    # get data
    data = get_data_from_database(
        database=args.regexSource,
        query_index=queries.get(args.regexGroup),
        **kwargs
    )

    logger.info(f'Work with <{len(data)}> samples')

    # init objects for similarity measuring
    sp = SreParser()
    ct = CustomTranslator()

    # get string regexes
    str_regexes = [x[0] for x in data]

    # parsing errors may occur when obtaining regular expression graphs.

    # get graph regexes by SRE parser
    new_str_regexes = []
    sre_regexes = []
    errors = 0
    for i, x in enumerate(str_regexes):
        try:
            sre_regex = sp(x)
        except Exception as e:
            logger.warning(
                f'<{errors}> Wrong regular expression parsing with SRE Parser.'
                f'\n\tRegex: <{x}>.'
                f'\n\tError: <{e}>.'
                f'\nTraceback: {traceback.format_exc()}'
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
                f'<{errors}> Wrong regular expression parsing with Custom Translator.'
                f'\n\tRegex: <{x}>.'
                f'\n\tError: <{e}>.'
            )
            errors += 1
            continue
        translator_regexes.append(translator_regex)
        new_str_regexes.append(str_regexes[i])
        new_sre_regexes.append(sre_regexes[i])
    str_regexes = new_str_regexes
    sre_regexes = new_sre_regexes

    logger.info(f'After parsing work with <{len(str_regexes)}> samples')

    # Regex as a string

    # Levenshtein
    logger.info('Levenshtein')
    l_res = get_similarity_matrix(
        name='levenshtein',
        regexes=str_regexes,
        similarity_func=StringSimilarity.get_distance,
    )

    # Hamming
    logger.info('Hamming')
    h_res = get_similarity_matrix(
        name='hamming',
        regexes=str_regexes,
        similarity_func=StringSimilarity.get_hamming_distance,
    )

    # Jaro
    logger.info('Jaro')
    j_res = get_similarity_matrix(
        name='jaro',
        regexes=str_regexes,
        similarity_func=StringSimilarity.get_jaro_similarity,
    )

    # Jaro-Winkler
    logger.info('Jaro-Winkler')
    jw_res = get_similarity_matrix(
        name='jaro_winkler',
        regexes=str_regexes,
        similarity_func=StringSimilarity.get_jaro_similarity,
        kwargs_func={'is_jaro_winkler': True}
    )

    # Regex as a graph

    # GED for SRE Parser
    logger.info('GED for SRE Parser')
    sre_res = get_similarity_matrix(
        name='sre',
        regexes=sre_regexes,
        similarity_func=GraphSimilarity.get_graph_edit_distance,
        kwargs_func={'is_optimize': True}
    )

    # GED for Custom Translator
    logger.info('GED for Custom Translator')
    translator_res = get_similarity_matrix(
        name='custom',
        regexes=translator_regexes,
        similarity_func=GraphSimilarity.get_graph_edit_distance,
        kwargs_func={'is_optimize': True}
    )
