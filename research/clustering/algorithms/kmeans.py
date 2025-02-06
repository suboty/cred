import time
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score


class KMeansAlgorithm:
    def __init__(
            self,
            max_number_of_clusters: int = 11,
            random_state: int = 42,
    ):
        self.random_state = random_state
        self.max_number_of_clusters = max_number_of_clusters

    def __call__(self, data, pipeline_name, *args, **kwargs):

        sse = []
        km_silhouette = []
        db_score = []

        for k in range(2, self.max_number_of_clusters):
            print('-'*10, f"{k} clusters", '-'*10)
            t0 = time.time()
            kmeans = KMeans(
                n_clusters=k,
                init='k-means++',
                random_state=self.random_state
            )
            kmeans.fit(data)
            sse.append(kmeans.inertia_)
            print(f"Score for number of cluster(s) {k}: {kmeans.inertia_}")

            preds = kmeans.predict(data)

            silhouette = silhouette_score(data, preds)
            km_silhouette.append(silhouette)
            print(f"Silhouette score for number of cluster(s) {k}: {silhouette}")

            db = davies_bouldin_score(data, preds)
            db_score.append(db)
            print(f"Davies Bouldin score for number of cluster(s) {k}: {db}")
            print(f'--- Elapsed time: {round(time.time()-t0)} seconds')

        # kmeans score (elbow method) saving
        _, ax = plt.subplots()
        ax.plot(range(2, self.max_number_of_clusters), sse, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Kmeans score')
        if 'tf_idf' in pipeline_name:
            ax.set_title(f"Kmeans (elbow method) score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
            ax.figure.savefig(Path('assets', 'tf_idf', f"{pipeline_name}_elbow.png"))
        else:
            raise NotImplementedError

        # silhouette saving
        _, ax = plt.subplots()
        ax.plot(range(2, self.max_number_of_clusters), km_silhouette, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Silhouette score')
        if 'tf_idf' in pipeline_name:
            ax.set_title(f"Kmeans silhouette score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
            ax.figure.savefig(Path('assets', 'tf_idf', f"{pipeline_name}_silhouette.png"))
        else:
            raise NotImplementedError

        # davies bouldin saving
        _, ax = plt.subplots()
        ax.plot(range(2, self.max_number_of_clusters), db_score, marker='o')
        ax.set_xlabel('Number of clusters')
        ax.set_ylabel('Davies Bouldin score')
        if 'tf_idf' in pipeline_name:
            ax.set_title(f"Kmeans Davies Bouldin score {pipeline_name.replace('tf_idf_', '')} for TF-IDF matrix")
            ax.figure.savefig(Path('assets', 'tf_idf', f"{pipeline_name}_db.png"))
        else:
            raise NotImplementedError
