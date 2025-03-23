import argparse
import warnings

import yaml

from utils import *
from logger import logger
from encoders.get_tf_idf_matrix import TfidfMatrix
from encoders.get_bert_embeddings import BertEmbeddings
from get_data import get_data_from_regex101
from algorithms.kmeans import KMeansAlgorithm
from preprocessing import Replacements


def load_yml_config(
    path_to_config=Path('algorithms.yml')
):
    with open(path_to_config) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as e:
            print(f'Error while parsing yaml config: {e}')


def iter_tf_idf(methods_list, **kwargs):
    if methods_list:
        for method_name in methods_list:
            match method_name:
                case 'tokens':
                    get_matrix_function = TfidfMatrix.get_matrix_tokenize_by_regex_tokens
                case 'chars':
                    get_matrix_function = TfidfMatrix.get_matrix_tokenize_by_chars
                case 'non_terminals':
                    get_matrix_function = TfidfMatrix.get_matrix_tokenize_by_non_terminals
                case _:
                    raise NotImplementedError

            run_tf_idf(
                tf_idf_method=method_name,
                get_matrix_function=get_matrix_function,
                **kwargs
            )


def iter_bert(methods_list, **kwargs):
    if methods_list:
        for method_name in methods_list:
            _be = BertEmbeddings(method_name)
            run_bert(
                _be=_be,
                **kwargs
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='cred-clustering')

    # is need verbose print
    parser.add_argument(
        '-v', '--verbose',
        action='store_true'
    )
    # is need update of visualization
    parser.add_argument(
        '-u', '--update',
        action='store_true'
    )
    # is need equivalent replacements in datasets
    parser.add_argument(
        '-e', '--equivalent',
        action='store_true'
    )
    # is need nearly equivalent replacements in datasets
    parser.add_argument(
        '-n', '--nearly-equivalent',
        action='store_true'
    )
    # encoder
    parser.add_argument('--algname', type=str, default='bert')
    # filter word for getting data
    parser.add_argument('--filter', type=str, default=None)

    # init
    args = parser.parse_args()

    km = KMeansAlgorithm()

    repl = Replacements()

    os.makedirs(Path('tmp', 'clustering_reports'), exist_ok=True)
    os.makedirs(Path('tmp', 'clusters'), exist_ok=True)

    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    data, labels = get_data_from_regex101(args.filter)

    logger.info(f'Work with {len(data)} samples')

    # get data
    dataset = pd.DataFrame(data, columns=labels)
    dataset = dataset.loc[dataset['regex'] != '']

    list_of_regexes = dataset['regex'].tolist()

    pre_list_of_regexes = repl(
        regex_list=list_of_regexes,
        need_equivalent=args.equivalent,
        need_nearly_equivalent=args.nearly_equivalent
    )

    dialects = dataset['dialect'].tolist()

    # random number for example printing
    random_n = random.randint(0, len(list_of_regexes))

    if args.verbose:
        logger.info(f'Example of regexes:')
        [
            logger.info(f"\t{i+1}) "+x)
            for i, x
            in enumerate(random.sample(list_of_regexes, 3))
        ]

    alg_config = load_yml_config()

    # run clustering
    if 'tf_idf' in args.algname:
        iter_tf_idf(
            methods_list=alg_config.get('tf_idf'),
            list_of_regexes=list_of_regexes,
            pre_list_of_regexes=pre_list_of_regexes,
            _verbose=args.verbose,
            random_keywords_number=random_n,
            _dialects=dialects,
            km_object=km,
            _filter=args.filter,
        )

        prepare_silh_table('tf_idf')

    if 'bert' in args.algname:
        iter_bert(
            methods_list=alg_config.get('bert'),
            _list_of_regexes=list_of_regexes,
            _pre_list_of_regexes=pre_list_of_regexes,
            _filter=args.filter,
            _dialects=dialects,
            _km=km,
        )

        prepare_silh_table('bert')
