import os
import time
from pathlib import Path
from typing import Union, List

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

from logger import logger
from algorithms.silhouette_analysis import make_silhouette_analysis
from db import (
    Experiments,
    Clustering,
    Results,
)


class KMeansAlgorithm:
    def __init__(
            self,
            max_number_of_clusters: int = 11,
            random_state: int = 42,
            excluded_metrics: List = None,
            cluster_step: int = 1,
            cluster_start: int = 2
    ):
        if excluded_metrics is None:
            excluded_metrics = ['db', 'elbow']
        self.random_state = random_state
        self.max_number_of_clusters = max_number_of_clusters
        self.cluster_step = cluster_step
        self.cluster_start = cluster_start
        self.excluded_metrics = excluded_metrics

    def __call__(
            self,
            data,
            pipeline_name,
            savepath: Union[str, Path],
            verbose: bool,
            data_2d,
            db,
            *args, **kwargs,
    ):

        clustered_results = {
            'original_data': {},
            '2d_data': {}
        }

        for i_data, _data in [(0, data), (1, data_2d)]:

            sse = []
            km_silhouette = []
            db_scores = []

            _cluster_range = list(
                range(
                    self.cluster_start,
                    self.max_number_of_clusters,
                    self.cluster_step
                )
            )

            if self.cluster_start > 2:
                _cluster_range.append(2)
                _cluster_range = sorted(_cluster_range)

            if self.cluster_step > 1:
                _cluster_range.append(3)
                _cluster_range.append(4)
                _cluster_range = list(set(_cluster_range))
                _cluster_range = sorted(_cluster_range)

            if verbose:
                logger.info(
                    f'Work with cluster range: '
                    f'{_cluster_range}'
                )

            for k in _cluster_range:

                exp_meta = db.create_experiment(
                    Experiments(
                        vectorizer=kwargs.get('_vectorizer'),
                        filter_word=kwargs.get('_filter'),
                        clustering_algorithm='kmeans++',
                        cluster_number=k,
                        preprocessed=kwargs.get('_preprocessed'),
                        input_data_shape='original' if i_data == 0 else '2d'
                    )
                )

                if verbose:
                    logger.debug('-' * 10, f"{k} clusters", '-' * 10)
                t0 = time.time()
                kmeans = KMeans(
                    n_clusters=k,
                    init='k-means++',
                    random_state=self.random_state
                )

                _data = np.squeeze(_data)

                kmeans.fit(_data)
                preds = kmeans.predict(_data)

                for i, _id in enumerate(kwargs.get('ids')):
                    db.create_clustering(
                        Clustering(
                            regex_id=_id,
                            experiment_id=exp_meta.get('id'),
                            cluster_id=int(preds[i])
                        )
                    )
                if os.environ.get('isRegexesSaving').lower() == 'true':
                    match i_data:
                        case 0:
                            clustered_results['original_data'][k] = preds
                        case 1:
                            clustered_results['2d_data'][k] = preds

                if 'elbow' not in self.excluded_metrics:
                    db.create_result(
                        Results(
                            experiment_id=exp_meta.get('id'),
                            metric_name='elbow',
                            metric_value=kmeans.inertia_
                        )
                    )
                    sse.append(kmeans.inertia_)
                    if verbose:
                        logger.debug(f"Score for number of cluster(s) {k}: {kmeans.inertia_}")

                if 'silhouette' not in self.excluded_metrics:
                    silhouette = silhouette_score(_data, preds)
                    db.create_result(
                        Results(
                            experiment_id=exp_meta.get('id'),
                            metric_name='silhouette',
                            metric_value=silhouette
                        )
                    )
                    km_silhouette.append(silhouette)
                    if verbose:
                        logger.debug(f"Silhouette score for number of cluster(s) {k}: {silhouette}")

                    if os.environ.get('isAssetsSaving').lower() == 'true':
                        make_silhouette_analysis(
                            X=data_2d,
                            clusterer=kmeans,
                            savepath=savepath,
                            cluster_labels=preds,
                            n_clusters=k,
                            pipeline_name=pipeline_name,
                            tip=None if i_data == 0 else '_data_2d'
                        )

                if 'db' not in self.excluded_metrics:
                    db_score = davies_bouldin_score(_data, preds)
                    db.create_result(
                        Results(
                            experiment_id=exp_meta.get('id'),
                            metric_name='davies bouldin',
                            metric_value=db
                        )
                    )
                    db_scores.append(db_score)
                    if verbose:
                        logger.debug(f"Davies Bouldin score for number of cluster(s) {k}: {db}")

                if verbose:
                    logger.debug(f'--- Elapsed time: {round(time.time() - t0)} seconds')

            if os.environ.get('isAssetsSaving').lower() == 'true':
                # kmeans score (elbow method) saving
                if 'elbow' not in self.excluded_metrics:
                    _, ax = plt.subplots()
                    ax.plot(range(
                        self.cluster_start, self.max_number_of_clusters, self.cluster_step
                    ), sse, marker='o')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Kmeans score')
                    if 'tf_idf' in pipeline_name:
                        ax.set_title(
                            f"Kmeans (elbow method) score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_elbow.png"))
                    elif 'bert' in pipeline_name:
                        ax.set_title(
                            f"Kmeans (elbow method) score {pipeline_name.replace('bert_', '')} for BERT embeddings"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_elbow.png"))
                    else:
                        raise NotImplementedError

                # silhouette saving
                if 'silhouette' not in self.excluded_metrics:
                    _, ax = plt.subplots()
                    ax.plot(_cluster_range, km_silhouette, marker='o')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Silhouette score')
                    if 'tf_idf' in pipeline_name:
                        ax.set_title(
                            f"Kmeans silhouette score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_silhouette.png"))
                    elif 'bert' in pipeline_name:
                        ax.set_title(
                            f"Kmeans silhouette score {pipeline_name.replace('bert_', '')} for BERT embeddings"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_silhouette.png"))
                    else:
                        raise NotImplementedError

                # davies bouldin saving
                if 'db' not in self.excluded_metrics:
                    _, ax = plt.subplots()
                    ax.plot(_cluster_range, db_scores, marker='o')
                    ax.set_xlabel('Number of clusters')
                    ax.set_ylabel('Davies Bouldin score')
                    if 'tf_idf' in pipeline_name:
                        ax.set_title(
                            f"Kmeans Davies Bouldin score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_db.png"))
                    elif 'bert' in pipeline_name:
                        ax.set_title(
                            f"Kmeans Davies Bouldin score {pipeline_name.replace('bert_', '')} for BERT embeddings"
                        )
                        ax.figure.savefig(Path(savepath, f"{pipeline_name}_db.png"))
                    else:
                        raise NotImplementedError

        return clustered_results
