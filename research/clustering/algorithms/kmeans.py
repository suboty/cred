import time
from pathlib import Path
from typing import List, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


class KMeansAlgorithm:
    def __init__(
            self,
            max_number_of_clusters: int = 11,
            random_state: int = 42,
            excluded_metrics: List = ['db', 'elbow'],
    ):
        self.random_state = random_state
        self.max_number_of_clusters = max_number_of_clusters
        self.excluded_metrics = excluded_metrics

    def __call__(
        self, 
        data, 
        pipeline_name,
        savepath: Union[str, Path],
        verbose: bool,
        *args, **kwargs,
    ):

        data = np.squeeze(data)

        sse = []
        km_silhouette = []
        db_score = []

        for k in range(2, self.max_number_of_clusters):
            if verbose: print('-'*10, f"{k} clusters", '-'*10)
            t0 = time.time()
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=self.random_state
            )
            kmeans.fit(data)

            if 'elbow' not in self.excluded_metrics:
                sse.append(kmeans.inertia_)
                if verbose: print(f"Score for number of cluster(s) {k}: {kmeans.inertia_}")

            preds = kmeans.predict(data)

            if 'silhouette' not in self.excluded_metrics:
                silhouette = silhouette_score(data, preds)
                km_silhouette.append(silhouette)
                if verbose: print(f"Silhouette score for number of cluster(s) {k}: {silhouette}")

            if 'db' not in self.excluded_metrics:
                db = davies_bouldin_score(data, preds)
                db_score.append(db)
                if verbose: print(f"Davies Bouldin score for number of cluster(s) {k}: {db}")

            if verbose:
                print(f'--- Elapsed time: {round(time.time()-t0)} seconds')

        # kmeans score (elbow method) saving
        if 'elbow' not in self.excluded_metrics:
            _, ax = plt.subplots()
            ax.plot(range(2, self.max_number_of_clusters), sse, marker='o')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Kmeans score')
            if 'tf_idf' in pipeline_name:
                ax.set_title(f"Kmeans (elbow method) score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_elbow.png"))
            elif 'bert' in pipeline_name:
                ax.set_title(f"Kmeans (elbow method) score {pipeline_name.replace('bert_', '')} for BERT embeddings")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_elbow.png"))
            else:
                raise NotImplementedError

        # silhouette saving
        if 'silhouette' not in self.excluded_metrics:
            _, ax = plt.subplots()
            ax.plot(range(2, self.max_number_of_clusters), km_silhouette, marker='o')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Silhouette score')
            if 'tf_idf' in pipeline_name:
                ax.set_title(f"Kmeans silhouette score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_silhouette.png"))
            elif 'bert' in pipeline_name:
                ax.set_title(f"Kmeans silhouette score {pipeline_name.replace('bert_', '')} for BERT embeddings")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_silhouette.png"))
            else:
                raise NotImplementedError

        # davies bouldin saving
        if 'db' not in self.excluded_metrics:
            _, ax = plt.subplots()
            ax.plot(range(2, self.max_number_of_clusters), db_score, marker='o')
            ax.set_xlabel('Number of clusters')
            ax.set_ylabel('Davies Bouldin score')
            if 'tf_idf' in pipeline_name:
                ax.set_title(f"Kmeans Davies Bouldin score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_db.png"))
            elif 'bert' in pipeline_name:
                ax.set_title(f"Kmeans Davies Bouldin score {pipeline_name.replace('bert_', '')} for BERT embeddings")
                ax.figure.savefig(Path(savepath, f"{pipeline_name}_db.png"))
            else:
                raise NotImplementedError
