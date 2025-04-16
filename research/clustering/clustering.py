import argparse
import warnings

import yaml

from utils import *
from logger import logger
from encoders.get_tf_idf_matrix import TfidfMatrix
from encoders.get_bert_embeddings import BertEmbeddings
from preprocessing.get_regex_ast import SreParser
from algorithms.kmeans import KMeansAlgorithm
from algorithms.cmeans import CMeansAlgorithm
from preprocessing import Replacements
from get_data import (
    get_data_from_regex101,
    get_data_from_regexlib,
    data_to_db
)
from db import (
    EntityMeta,
    ResearchRepository
)


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
    # is need scaling for embeddings
    parser.add_argument(
        '-s', '--scaling',
        action='store_true'
    )
    os.environ['IS_NEED_SCALING'] = 'true'
    # encoder
    parser.add_argument('--algname', type=str, default='bert')
    # filter word for getting data
    parser.add_argument('--filter', type=str, default=None)
    # clusters number
    parser.add_argument('--clustersnum', type=int, default=10)
    # clusters step
    parser.add_argument('--clusterstep', type=int, default=1)
    # clusters start
    parser.add_argument('--clusterstart', type=int, default=2)
    # regex source
    parser.add_argument('--source', type=str, default='regex101')
    # clustering algorithm
    parser.add_argument('--clusteringname', type=str, default='kmeans')

    # init objects
    args = parser.parse_args()

    match args.clusteringname:
        case 'kmeans':
            cluster_alg = KMeansAlgorithm(
                max_number_of_clusters=args.clustersnum + 1,
                cluster_start=args.clusterstart,
                cluster_step=args.clusterstep
            )
        case 'cmeans':
            cluster_alg = CMeansAlgorithm(
                max_number_of_clusters=args.clustersnum + 1,
                cluster_start=args.clusterstart,
                cluster_step=args.clusterstep
            )
        case _:
            raise NotImplementedError

    repl = Replacements()
    parser = SreParser()

    # prepare folders
    os.makedirs(Path('tmp', 'clustering_reports'), exist_ok=True)
    os.makedirs(Path('tmp', 'clusters'), exist_ok=True)

    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    # init databases
    db = ResearchRepository(
        database_url=f'sqlite:///tmp/research.db',
        entity_meta=EntityMeta,
    )

    # get data
    match args.source:
        case 'regex101':
            data, columns = get_data_from_regex101(args.filter)
            label_column = 'dialect'
            regex_column = 'regex'
        case 'regexlib':
            data, columns = get_data_from_regexlib(args.filter)
            label_column = 'rating'
            regex_column = 'pattern'
        case _:
            raise NotImplementedError

    if 'bert' in args.algname and args.filter == '_':
        # TODO: fix memory error
        if len(data) > 5000:
            indexes = np.random.randint(0, len(data)-1, 5000)
            new_data = []
            for index in indexes:
                new_data.append(data[index])
            data = new_data

    logger.info(f'Work with {len(data)} samples')

    dataset = pd.DataFrame(data, columns=columns)
    dataset = dataset.loc[dataset[regex_column] != '']

    try:
        labels = [
            str(x) for x in
            dataset[label_column].tolist()
        ]
    except Exception as e:
        logger.error(f'This dataset has no labels! Error: {e}')
        exit(1)

    # 1 (original regexes)
    list_of_regexes = dataset[regex_column].tolist()
    ids = data_to_db(
        db=db,
        regexes=list_of_regexes,
        labels=labels
    )

    # 2 (preprocessing regexes)
    pre_list_of_regexes = repl(
        regex_list=list_of_regexes,
        need_equivalent=args.equivalent,
        need_nearly_equivalent=args.nearly_equivalent
    )
    pre_ids = data_to_db(
        db=db,
        regexes=pre_list_of_regexes,
        labels=labels,
        is_preprocessed=True,
    )

    # 3 (ast for original regexes)
    ast_regex, ast_labels = parser.parse_list(
        regex_list=list_of_regexes,
        dialects=labels
    )
    ast_ids = data_to_db(
        db=db,
        regexes=ast_regex,
        labels=ast_labels,
        is_ast=True,
    )

    # 4 (ast for preprocessing regexes)
    pre_ast_regex, pre_ast_labels = parser.parse_list(
        regex_list=pre_list_of_regexes,
        dialects=labels
    )
    pre_ast_ids = data_to_db(
        db=db,
        regexes=pre_ast_regex,
        labels=pre_ast_labels,
        is_ast=True,
        is_preprocessed=True,
    )

    # prepare data tuple
    input_data = (
        # data | labels | tip | ids
        (list_of_regexes, labels, 'original', ids),
        (pre_list_of_regexes, labels, 'pre', pre_ids),
        (ast_regex, ast_labels, 'ast_original', ast_ids),
        (pre_ast_regex, pre_ast_labels, 'ast_pre', pre_ast_ids),
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
    if 'tf_idf' in args.algname and alg_config.get('tf_idf'):
        iter_tf_idf(
            methods_list=alg_config.get('tf_idf'),
            input_data=input_data,
            _verbose=args.verbose,
            random_keywords_number=random_n,
            km_object=cluster_alg,
            _filter=args.filter,
            db=db,
        )
        try:
            prepare_silh_table(
                tip='tf-idf',
                filter_word=args.filter,
            )
        except Exception as e:
            logger.warning(f'Error while stats saving: {e}')

    if 'bert' in args.algname and alg_config.get('bert'):
        iter_bert(
            methods_list=alg_config.get('bert'),
            input_data=input_data,
            _filter=args.filter,
            _km=cluster_alg,
            db=db,
        )
        try:
            prepare_silh_table(
                tip='bert',
                filter_word=args.filter,
            )
        except Exception as e:
            logger.warning(f'Error while stats saving: {e}')

    prepare_results(
        filter_word=args.filter,
    )
