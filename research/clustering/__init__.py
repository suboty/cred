import os
import time
import warnings
from typing import List, Optional
from pathlib import Path

from sqlalchemy import text

from clustering import load_yml_config
from algorithms.kmeans import KMeansAlgorithm
from algorithms.cmeans import CMeansAlgorithm
from preprocessing import Replacements
from preprocessing.get_regex_ast import SreParser
from get_data import data_to_db
from logger import logger
from clustering import iter_tf_idf, iter_bert
from db import (
    EntityMeta,
    ResearchRepository
)


class ClusteringError(Exception):
    ...


class ClusteringUseCase:
    def __init__(
            self,
            path_to_algorithms_yaml: Path,
            path_to_preprocessing: Path,
            path_to_encoders: Path,
            path_to_sql_queries: Path,
            embedding_scaling: bool = True,
    ):
        self.creating_time = int(time.time())

        self.algorithms = load_yml_config(path_to_algorithms_yaml)
        self.path_to_preprocessing = path_to_preprocessing
        self.path_to_encoders = path_to_encoders

        # load sql queries for result
        self.queries = {}
        with open(Path(path_to_sql_queries, 'get_best_experiment_id.sql'), 'r') as b_exp_file:
            data = b_exp_file.read()
            data.replace('TIMEID', str(self.creating_time))
            self.queries['best_id'] = data
        with open(Path(path_to_sql_queries, 'get_clustering_regexes.sql'), 'r') as c_reg_file:
            data = c_reg_file.read()
            data.replace('TIMEID', str(self.creating_time))
            self.queries['clust_reg'] = data

        # work with system variables
        if embedding_scaling:
            os.environ['IS_NEED_SCALING'] = 'true'
        os.environ['isRegexesSaving'] = 'false'
        os.environ['isClusteringReportsSaving'] = 'false'
        os.environ['isAssetsSaving'] = 'false'

        # disable warnings from scikit-learn and umap-learn
        warnings.filterwarnings("ignore")

    def __call__(
        self,
        input_regexes: List[str],
        labels: Optional[List] = None,
        verbose: bool = False,
        equivalent: bool = True,
        nearly_equivalent: bool = True,
        cluster_num: int = 50,
        cluster_step: int = 1,
        cluster_start: int = 2,
        is_need_to_delete_tmp_db: bool = False,
    ):
        try:
            # Step 1: Init clustering algorithms and database
            kmeans = KMeansAlgorithm(
                max_number_of_clusters=cluster_num + 1,
                cluster_start=cluster_start,
                cluster_step=cluster_step
            )
            cmeans = CMeansAlgorithm(
                max_number_of_clusters=cluster_num + 1,
                cluster_start=cluster_start,
                cluster_step=cluster_step
            )
            tmp_database = ResearchRepository(
                database_url=f'sqlite:///clustering_{self.creating_time}.db',
                entity_meta=EntityMeta,
            )
            os.makedirs('tmp', exist_ok=True)

            # Step 2: Init objects for preprocessing
            repl = Replacements(self.path_to_preprocessing)
            parser = SreParser()

            # Step 3: Prepare data
            input_regexes = [x for x in input_regexes if x not in ('', None, ' ')]
            # Step 3.1: Add slug labels
            if not labels:
                # add slug labels
                labels = len(input_regexes)*'0'.split()

            # Step 3.2: Prepare original regexes
            list_of_regexes = list(input_regexes)
            ids = data_to_db(
                db=tmp_database,
                regexes=list_of_regexes,
                labels=labels
            )

            # Step 3.3: Prepare preprocessing regexes
            pre_list_of_regexes = repl(
                regex_list=list_of_regexes,
                need_equivalent=equivalent,
                need_nearly_equivalent=nearly_equivalent
            )
            pre_ids = data_to_db(
                db=tmp_database,
                regexes=pre_list_of_regexes,
                labels=labels,
                is_preprocessed=True,
            )

            # Step 3.4: Prepare ast for original regexes
            ast_regex, ast_labels = parser.parse_list(
                regex_list=list_of_regexes,
                dialects=labels
            )
            ast_ids = data_to_db(
                db=tmp_database,
                regexes=ast_regex,
                labels=ast_labels,
                is_ast=True,
            )

            # Step 3.4: Prepare ast for preprocessing regexes
            pre_ast_regex, pre_ast_labels = parser.parse_list(
                regex_list=pre_list_of_regexes,
                dialects=labels
            )
            pre_ast_ids = data_to_db(
                db=tmp_database,
                regexes=pre_ast_regex,
                labels=pre_ast_labels,
                is_ast=True,
                is_preprocessed=True,
            )

            # Step 3.5: Prepare data tuple
            input_data = (
                # data | labels | tip | ids
                (list_of_regexes, labels, 'original', ids),
                (pre_list_of_regexes, labels, 'pre', pre_ids),
                (ast_regex, ast_labels, 'ast_original', ast_ids),
                (pre_ast_regex, pre_ast_labels, 'ast_pre', pre_ast_ids),
            )

            # Step 4: Get clustering results for algorithms
            for clu_alg in [kmeans, cmeans]:
                if self.algorithms.get('tf_idf'):
                    iter_tf_idf(
                        path_to_encoders=self.path_to_encoders,
                        methods_list=self.algorithms.get('tf_idf'),
                        input_data=input_data,
                        _verbose=verbose,
                        km_object=clu_alg,
                        db=tmp_database,
                    )

                if self.algorithms.get('tf_idf'):
                    iter_bert(
                        methods_list=self.algorithms.get('bert'),
                        input_data=input_data,
                        _km=clu_alg,
                        db=tmp_database,
                    )

            # Step 5: Get clustering regexes by best algorithm
            db_engine = tmp_database.engine
            try:
                with db_engine.connect() as connection:

                    best_id = connection.execute(text(
                        self.queries.get('best_id')
                    )).fetchone()

                    clustering_regexes = connection.execute(text(
                        self.queries.get('clust_reg').replace('BESTEXPID', str(best_id))
                    )).fetchall()

            except Exception as e:
                logger.error('Error while database work!')
                raise e

            logger.info(
                f'Clustering is finish. Elapsed time: {round(time.time()-self.creating_time)} sec.'
            )
            return clustering_regexes
        except Exception as e:
            raise ClusteringError(f'Something wrong while clustering! Error: {e}')
