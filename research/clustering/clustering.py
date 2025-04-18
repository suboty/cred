import time
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
            res_yaml = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(f'Error while parsing yaml config: {exc}')

    # for statistics calculate
    if res_yaml.get('tf_idf'):
        if 'tokens' not in res_yaml.get('tf_idf'):
            res_yaml['tf_idf'].append('tokens')
    else:
        res_yaml['tf_idf'] = ['tokens']
    return res_yaml


def iter_tf_idf(methods_list, **kwargs):
    if methods_list:
        for method_name in methods_list:
            tf_idf_object = TfidfMatrix(kwargs.get('path_to_encoders'))
            match method_name:
                case 'tokens':
                    get_matrix_function = tf_idf_object.get_matrix_tokenize_by_regex_tokens
                case 'chars':
                    get_matrix_function = tf_idf_object.get_matrix_tokenize_by_chars
                case 'non_terminals':
                    get_matrix_function = tf_idf_object.get_matrix_tokenize_by_non_terminals
                case _:
                    raise NotImplementedError

            t0 = time.time()
            run_tf_idf(
                tf_idf_method=method_name,
                get_matrix_function=get_matrix_function,
                **kwargs
            )
            logger.info(f'Elapsed time: {round(time.time()-t0, 2)}')


def iter_bert(methods_list, **kwargs):
    if methods_list:
        for method_name in methods_list:
            _be = BertEmbeddings(method_name)
            t0 = time.time()
            run_bert(
                _be=_be,
                **kwargs
            )
            logger.info(f'Elapsed time: {round(time.time() - t0, 2)}')


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
    parser.add_argument('--algName', type=str, default='bert')
    # filter word for getting data
    parser.add_argument('--filter', type=str, default=None)
    # clusters number
    parser.add_argument('--clustersNum', type=int, default=10)
    # clusters step
    parser.add_argument('--clusterStep', type=int, default=1)
    # clusters start
    parser.add_argument('--clusterStart', type=int, default=2)
    # regex source
    parser.add_argument('--regexSource', type=str, default='regex101')
    # clustering algorithm
    parser.add_argument('--clusteringName', type=str, default='kmeans')
    # is need save clustering reports
    parser.add_argument('--isClusteringReportsSaving', type=str, default='y')
    # is need save regexes after clustering
    parser.add_argument('--isRegexesSaving', type=str, default='y')
    # is need save assets after clustering
    parser.add_argument('--isAssetsSaving', type=str, default='y')

    # init objects
    args = parser.parse_args()

    if args.isClusteringReportsSaving.lower() == 'y':
        os.environ['isClusteringReportsSaving'] = 'true'
    else:
        os.environ['isClusteringReportsSaving'] = 'false'

    if args.isAssetsSaving.lower() == 'y':
        os.environ['isAssetsSaving'] = 'true'
    else:
        os.environ['isAssetsSaving'] = 'false'

    if args.isRegexesSaving.lower() == 'y':
        os.environ['isRegexesSaving'] = 'true'
    else:
        os.environ['isRegexesSaving'] = 'false'

    match args.clusteringName:
        case 'kmeans':
            cluster_alg = KMeansAlgorithm(
                max_number_of_clusters=args.clustersNum + 1,
                cluster_start=args.clusterStart,
                cluster_step=args.clusterStep
            )
        case 'cmeans':
            cluster_alg = CMeansAlgorithm(
                max_number_of_clusters=args.clustersNum + 1,
                cluster_start=args.clusterStart,
                cluster_step=args.clusterStep
            )
        case _:
            raise NotImplementedError

    repl = Replacements()
    parser = SreParser()

    # prepare folders
    os.makedirs('tmp', exist_ok=True)
    if os.environ.get('isClusteringReportsSaving').lower() == 'true':
        os.makedirs(Path('tmp', 'clustering_reports'), exist_ok=True)
    if os.environ.get('isRegexesSaving').lower() == 'true':
        os.makedirs(Path('tmp', 'clusters'), exist_ok=True)

    # disable warnings from scikit-learn and umap-learn
    warnings.filterwarnings("ignore")

    # init databases
    db = ResearchRepository(
        database_url=f'sqlite:///tmp/research.db',
        entity_meta=EntityMeta,
    )

    # get data
    match args.regexSource:
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

    if 'bert' in args.algName and args.filter == '_':
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
    if 'tf_idf' in args.algName and alg_config.get('tf_idf'):
        iter_tf_idf(
            methods_list=alg_config.get('tf_idf'),
            input_data=input_data,
            _verbose=args.verbose,
            random_keywords_number=random_n,
            km_object=cluster_alg,
            _filter=args.filter,
            db=db,
        )
        if os.environ.get('isClusteringReportsSaving').lower() == 'true':
            try:
                prepare_silh_table(
                    tip='tf-idf',
                    filter_word=args.filter,
                )
            except Exception as e:
                logger.warning(f'Error while stats saving: {e}')

    if 'bert' in args.algName and alg_config.get('bert'):
        iter_bert(
            methods_list=alg_config.get('bert'),
            input_data=input_data,
            _filter=args.filter,
            _km=cluster_alg,
            db=db,
        )
        if os.environ.get('isClusteringReportsSaving').lower() == 'true':
            try:
                prepare_silh_table(
                    tip='bert',
                    filter_word=args.filter,
                )
            except Exception as e:
                logger.warning(f'Error while stats saving: {e}')
    if os.environ.get('isClusteringReportsSaving').lower() == 'true':
        prepare_results(
            filter_word=args.filter,
        )
