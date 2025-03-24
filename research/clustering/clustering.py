import argparse
import warnings

import yaml

from utils import *
from logger import logger
from encoders.get_tf_idf_matrix import TfidfMatrix
from encoders.get_bert_embeddings import BertEmbeddings
from preprocessing.get_regex_ast import SreParser
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

    # init objects
    args = parser.parse_args()
    km = KMeansAlgorithm()
    repl = Replacements()
    parser = SreParser()

    # prepare folders
    os.makedirs(Path('tmp', 'clustering_reports'), exist_ok=True)
    os.makedirs(Path('tmp', 'clusters'), exist_ok=True)

    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    # get data

    data, labels = get_data_from_regex101(args.filter)
    logger.info(f'Work with {len(data)} samples')

    dataset = pd.DataFrame(data, columns=labels)
    dataset = dataset.loc[dataset['regex'] != '']

    try:
        labels = dataset['dialect'].tolist()
    except Exception as e:
        logger.error(f'This dataset has no labels! Error: {e}')
        exit(1)

    # 1 (original regexes)
    list_of_regexes = dataset['regex'].tolist()

    # 2 (preprocessing regexes)
    pre_list_of_regexes = repl(
        regex_list=list_of_regexes,
        need_equivalent=args.equivalent,
        need_nearly_equivalent=args.nearly_equivalent
    )

    # 3 (ast for original regexes)
    ast_regex, ast_labels = parser.parse_list(
        regex_list=list_of_regexes,
        dialects=labels
    )

    # 4 (ast for preprocessing regexes)
    pre_ast_regex, pre_ast_labels = parser.parse_list(
        regex_list=pre_list_of_regexes,
        dialects=labels
    )

    # prepare data tuple
    input_data = (
        # data | labels | tip
        (list_of_regexes, labels, 'original'),
        (pre_list_of_regexes, labels, 'pre'),
        (ast_regex, ast_labels, 'ast_original'),
        (pre_ast_regex, pre_ast_labels, 'ast_pre'),
    )

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
            input_data=input_data,
            _verbose=args.verbose,
            random_keywords_number=random_n,
            km_object=km,
            _filter=args.filter,
        )

        prepare_silh_table(
            tip='tf-idf',
            filter_word=args.filter,
        )

    if 'bert' in args.algname:
        iter_bert(
            methods_list=alg_config.get('bert'),
            input_data=input_data,
            _filter=args.filter,
            _km=km,
        )

        prepare_silh_table(
            tip='bert',
            filter_word=args.filter,
        )

    prepare_results(
        filter_word=args.filter,
    )
