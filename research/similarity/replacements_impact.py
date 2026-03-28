import warnings
from pathlib import Path

from tqdm import tqdm

from logger import logger
from get_data import get_data_from_database
from generator import Generator
from generator.get_replacements_by_sre import generate
from preprocessing.sre import SreParser
from preprocessing.custom import CustomTranslator


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    # get data
    data = get_data_from_database(
        database='regex101',
        query_index=4,
    )

    generate(
        [x[0] for x in data],
        min_frequency=5,
        logger=logger,
        path=Path('generator', 'replacements')
    )

    sp = SreParser()
    ct = CustomTranslator()
    g = Generator()

    # get dialects and regexes
    regexes = [x[0] for x in data]
    dialects = []
    original_regexes = []
    repl_regexes = []

    repl_regexes_dict = g(
        regex_list=regexes
    )
    for i, regex in enumerate(repl_regexes_dict.keys()):
        if repl_regexes_dict[regex]:
            dialects.append(data[i][1])
            original_regexes.append(regex)
            repl_regexes.append(repl_regexes_dict[regex])

    logger.info(f'Work with <{len(dialects)}> samples')

    sp_results = []
    ct_results = []

    for i, dialect in enumerate(tqdm(dialects)):
        # Custom Translator
        try:
            ct(original_regexes[i])
        except:
            is_ok = False
            for repl in repl_regexes[i]:
                try:
                    ct(repl)
                    is_ok = True
                    break
                except:
                    pass
            ct_results.append((is_ok, dialect))

        # SRE Parser
        try:
            sp(original_regexes[i])
        except:
            is_ok = False
            for repl in repl_regexes[i]:
                try:
                    sp(repl)
                    is_ok = True
                    break
                except:
                    pass
            sp_results.append((is_ok, dialect))

    repl_success_ct = sum([x[0] for x in ct_results])
    repl_success_sp = sum([x[0] for x in sp_results])

    repl_failed_ct = len(ct_results) - repl_success_ct
    repl_failed_sp = len(sp_results) - repl_success_sp

    logger.warning(
        f'Replacements success with Custom Translator: {repl_success_ct}, '
        f'{round(repl_success_ct / len(ct_results), 2)*100}%'
    )

    logger.error(
        f'Replacements failed with Custom Translator: {repl_failed_ct}, '
        f'{round(repl_failed_ct / len(ct_results), 2) * 100}%'
    )

    logger.warning(
        f'Replacements success with SRE Parser: {repl_success_sp}, '
        f'{round(repl_success_sp / len(sp_results), 2) * 100}%'
    )

    logger.error(
        f'Replacements failed with SRE Parser: {repl_failed_sp}, '
        f'{round(repl_failed_sp / len(sp_results), 2) * 100}%'
    )
