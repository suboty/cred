import random
from pathlib import Path
from typing import Optional, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def get_tf_idf_keywords(
        _tfidf_vectorizer,
        _tfidf_matrix,
        document_index: Optional[int] = None,
        _list_of_regexes: Optional[List] = None
) -> None:
    """Get keywords in document."""
    if document_index is None:
        document_index = random.randint(0, len(_list_of_regexes))

    feature_names = _tfidf_vectorizer.get_feature_names_out()
    tfidf_scores = _tfidf_matrix.toarray()[document_index]

    # sort by TF-IDF values
    sorted_keywords = [
        (word, score)
        for score, word in sorted(
            zip(tfidf_scores, feature_names),
            reverse=True
        )
    ]

    print(f'\tExample regex: {_list_of_regexes[document_index]}')
    print(f"\tIts keywords:")
    i = 0
    for char, score in sorted_keywords:
        if score < 0.001 or i == 10:
            break
        print(f'\t--- {char} | {score}')
        i += 1


def get_all_unicode_letters(
        start_code: str,
        stop_code: str
) -> List[str]:
    """Get unicode letters by interval."""
    start_idx, stop_idx = [int(code, 16) for code in (start_code, stop_code)]
    characters = []
    for unicode_idx in range(start_idx, stop_idx + 1):
        characters.append(chr(unicode_idx))
    return characters


def high_dimensional_visualization(
        data: Union[List, np.array],
        name: str,
        dialects: List,
        umap_min_dist: float = 0.1,
        n_components: int = 2,
        n_neighbors: int = 100,
) -> None:
    """
    Function for visualization high-dimensional data.

    :param data: The features matrix from input regexes
    :param str name: The name of algorithm for getting features matrix
    :param int n_components: The dimension of reduced data space
    :param int n_neighbors: The number of neighbors for UMAP method
    :param float umap_min_dist: The minimal distance between points for UMAP method
    :param List dialects: The list of dialects of input regexes
    :type data: list or numpy array
    """
    plt.figure(figsize=(5, 5))

    rndprm = np.random.permutation(data.shape[0])

    # PCA visualization
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(data)

    df_pca = pd.DataFrame()
    df_pca['pca-one'] = pca_result[:, 0]
    df_pca['pca-two'] = pca_result[:, 1]

    df_pca['dialects'] = dialects
    emb_plot_pca = sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="dialects",
        data=df_pca.loc[rndprm, :],
        legend="full",
        alpha=0.3
    )
    emb_plot_pca.set_title(
        f"TF-IDF matrix by {name.replace('tf_idf_', '')} PCA visualization"
    )
    fig_pca = emb_plot_pca.get_figure()
    fig_pca.savefig(Path('assets', f"{name}_pca.png"))

    # UMAP visualization

    umap_method = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=umap_min_dist,
        n_components=n_components,
        metric='euclidean'
    )

    umap_result = umap_method.fit_transform(data)

    unique_dialects = {x: i for i, x in enumerate(list(set(dialects)))}
    ids_dialects = list(map(lambda x: unique_dialects[x], dialects))

    plt.scatter(
        umap_result[:, 0],
        umap_result[:, 1],
        c=ids_dialects,
        cmap='Spectral',
        s=5
    )
    plt.gca().set_aspect('equal', 'datalim')
    plt.title(f"TF-IDF matrix by {name.replace('tf_idf_', '')} UMAP visualization")
    plt.savefig(Path('assets', f"{name}_umap.png"))
