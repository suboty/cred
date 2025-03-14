import os
import json
import random
from pathlib import Path
from typing import Optional, List, Union, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import umap.umap_ as umap
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

IS_V2 = True


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
    if isinstance(_tfidf_matrix, (np.ndarray, np.generic)):
        tfidf_scores = _tfidf_matrix[document_index]
    else:
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


def get_table_from_values(_dict):
    _table = '<table>' \
             '<thead>' \
             '<tr><th>repl_str</th><th>Count</th></tr>' \
             '</thead>' \
             '<tbody>' \
             '__ROWS__' \
             '</tbody>' \
             '</table>'
    sim_rows = ''
    for key in _dict.keys():
        sim_rows += f'<tr><td>{key}</td><td>{_dict[key]}</td></tr>'
    return _table.replace('__ROWS__', sim_rows)


def high_dimensional_visualization(
        data: Union[List, np.array],
        name: str,
        dialects: List,
        savepath: Union[str, Path],
        umap_min_dist: float = 0.1,
        n_components: int = 2,
        n_neighbors: int = 100,
) -> Tuple:
    """
    Function for visualization high-dimensional data.

    :param data: The features matrix from input regexes
    :param str name: The name of algorithm for getting features matrix
    :param int n_components: The dimension of reduced data space
    :param int n_neighbors: The number of neighbors for UMAP method
    :param float umap_min_dist: The minimal distance between points for UMAP method
    :param List dialects: The list of dialects of input regexes
    :param Union[str, Path] savepath: path for saving plots
    :type data: list or numpy array
    """
    plt.figure(figsize=(5, 5))

    # Turn interactive plotting off
    plt.ioff()

    if isinstance(data, List):
        data = np.squeeze(np.array(data))

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
    if 'tf_idf' in name:
        emb_plot_pca.set_title(
            f"TF-IDF matrix by {name.replace('tf_idf_', '')} PCA visualization"
        )
        fig_pca = emb_plot_pca.get_figure()
        fig_pca.savefig(Path(savepath, f"{name}_pca.png"))
    elif 'bert' in name:
        emb_plot_pca.set_title(
            f"BERT embeddings by {name.replace('bert_', '')} PCA visualization"
        )
        fig_pca = emb_plot_pca.get_figure()
        fig_pca.savefig(Path(savepath, f"{name}_pca.png"))
    else:
        raise NotImplementedError

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
    if 'tf_idf' in name:
        plt.title(f"TF-IDF matrix by {name.replace('tf_idf_', '')} UMAP visualization")
        plt.savefig(Path(savepath, f"{name}_umap.png"))
    elif 'bert' in name:
        plt.title(f"BERT embeddings by {name.replace('tf_idf_', '')} UMAP visualization")
        plt.savefig(Path(savepath, f"{name}_umap.png"))
    else:
        raise NotImplementedError

    return pca_result, umap_result


def get_experiment_name(
        alg_name: str,
        filter_word: Optional[str] = None
):
    _name = f'exp_{alg_name}'
    if filter_word: _name += f"_{filter_word.lower().replace(' ', '_').replace('-', '_')}"
    return _name


def make_clustering_report(
        experiment_name: str,
        encoder: str,
        clustering: str,
        img_savepath: Union[str, Path],
        savepath: Union[str, Path],
        filter_word: Optional[str] = None,
        template_path: Path = Path('templates', 'template.html')
):
    # TODO: fix algorithm`s reports and variables

    if IS_V2:
        template_path = Path('templates', 'template_v2.html')

    with open(template_path, 'r') as template_file:
        template = template_file.read()

    # change experiment name
    template = template.replace('__EXPERIMENT_NAME__', experiment_name)

    # change encoder name
    template = template.replace('__ENCODER_NAME__', encoder)

    # change clustering algorithm name
    template = template.replace('__CLUSTERING_NAME__', clustering)

    # if filter word existing
    # change filter word in template
    if filter_word:
        template = template.replace('__FILTER_WORD__', filter_word)
    else:
        template = template.replace(
            '\t\t<p>\n\t\t\tFilter Word: <b>__FILTER_WORD__</b>\n\t\t</p>',
            ''
        )

    # change savepath (for plots)
    template = template.replace('__SAVEPATH__', str(img_savepath))

    # clustering analysis
    silh_plots = sorted([x for x in os.listdir(
        str(img_savepath).replace('..', 'tmp')
    ) if 'silh_' in x and 'pre_' not in x
                         ])
    pre_silh_plots = sorted([x for x in os.listdir(
        str(img_savepath).replace('..', 'tmp')
    ) if 'silh_' in x and 'pre_' in x
                             ])
    silh_vis = []
    for i, silh_plot in enumerate(silh_plots):
        _row = '\t\t<p float="left">'
        _row += f'<img src="{str(img_savepath)}/{silh_plot}" width="700" />'
        if len(pre_silh_plots) != 0:
            _row += f'<img src="{str(img_savepath)}/{pre_silh_plots[i]}" width="700" />'
        _row += '</p>'
        silh_vis.append(_row)
    silh_vis = '\n'.join(silh_vis)

    # add token stats table
    with open(Path('tmp', 'tokens_stats.json'), 'r') as st_f:
        token_stats = json.load(st_f)
    tokens_stats_table = '\n<table>' \
                         '\n\t<thead>' \
                         '\n\t\t<tr><th>Token</th><th>Count</th></tr>' \
                         '\n\t</thead>' \
                         '\n\t<tbody>' \
                         '\n\t\t__ROWS__' \
                         '\n\t</tbody>' \
                         '\n</table>'

    _rows = ''
    for key in token_stats.keys():
        _rows += f'<tr><td>{key}</td><td>{token_stats[key]}</td></tr>'

    tokens_stats_table = tokens_stats_table.replace('__ROWS__', _rows)
    if IS_V2:
        template = template.replace('__TOKENS_STATS_TABLE__', tokens_stats_table)

    template = template.replace('__CLUSTERS_VISUALISATION__', silh_vis)

    # add replacements statistics
    if 'pre' in experiment_name:
        with open(Path('tmp', 'replacements_stats.json'), 'r') as st_f:
            repl_stats = json.load(st_f)
        repl_stats_table = '\n<table>' \
                           '\n\t<thead>' \
                           '\n\t\t<tr><th>Replacement group</th><th>Replacement</th></tr>' \
                           '\n\t</thead>' \
                           '\n\t<tbody>' \
                           '\n\t\t__ROWS__' \
                           '\n\t</tbody>' \
                           '\n</table>'
        _rows = ''
        for key in repl_stats.keys():
            method_stats_table = '<table>' \
                                 '<thead>' \
                                 '<tr><th>Method</th><th>Count</th></tr>' \
                                 '</thead>' \
                                 '<tbody>' \
                                 '__ROWS__' \
                                 '</tbody>' \
                                 '</table>'
            m_rows = ''
            for method in repl_stats[key]:
               m_rows += f'<tr><td>{method}</td><td>' \
                         f'{get_table_from_values(repl_stats[key][method])}' \
                         f'</td></tr>'
            method_stats_table = method_stats_table.replace('__ROWS__', m_rows)

            _rows += f'<tr><td>{key}</td><td>{method_stats_table}</td></tr>'

        repl_stats_table = repl_stats_table.replace('__ROWS__', _rows)
        if IS_V2:
            template = template.replace('__REPLACEMENTS_STATS_TABLE', repl_stats_table)
    else:
        if IS_V2:
            template = template.replace('__REPLACEMENTS_STATS_TABLE', '')

    # create report
    with open(Path(savepath, f'{experiment_name}.html'), 'w') as report_file:
        report_file.write(template)
