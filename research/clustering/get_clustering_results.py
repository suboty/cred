import os
import re
import random
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


line_styles = [
    'solid',
    'dotted',
    'dashed',
    'dashdot'
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='cred-clustering')
    parser.add_argument('--pathresults', type=str, default='tmp')
    args = parser.parse_args()

    os.makedirs(
        'clustering_results_for_report',
        exist_ok=True
    )
    path_to_results = args.pathresults

    results = [
        x for x in
        os.listdir(path_to_results)
        if '.xlsx' in x and '_silhouette_results' in x
    ]

    reg_for_filter = re.compile(r'(?<=results_).*.(?=\.xlsx$)')

    exp = {}
    # 1 - data (filter word)
    #   2 - algorithm
    #       3 - is ast?
    #           4 - preprocess?
    #               5 - vector space
    #                   6 - cluster num

    for result in results:

        # get filter word
        filter_word = reg_for_filter.search(result).group(0)

        if not exp.get(filter_word):
            exp[filter_word] = {}

        data = pd.read_excel(
            Path(path_to_results, result),
            index_col=None
        )

        # preprocess data
        if len(data.columns) > 2:
            data = data.drop(columns=[data.columns[0]])

        data = data.apply(
            lambda row: list(
                (
                    *str(row.iloc[0]).split(' | '),
                    row.iloc[1]
                )
            ),
            axis=1
        )

        # process data
        for row in data:
            _exp = row[0]

            # get alg name
            if filter_word != '_':
                alg_name = _exp.replace(filter_word, '').replace('.xlsx', '')[:-1]
            else:
                alg_name = _exp.replace('.xlsx', '')[:-2]
            alg_name = alg_name.replace('_silhouette_results', '')

            if not exp[filter_word].get(alg_name):
                exp[filter_word][alg_name] = {}

            is_ast = 'ast' if 'ast' in _exp else 'regex'
            if not exp[filter_word][alg_name].get(is_ast):
                exp[
                    filter_word][
                    alg_name][
                    is_ast
                ] = {}

            is_pre = 'pre' if 'pre' in _exp else 'no_pre'
            if not exp[
                filter_word][
                alg_name][
                is_ast
            ].get(is_pre):
                exp[
                    filter_word][
                    alg_name][
                    is_ast][
                    is_pre
                ] = {}

            _data = row[2]
            if not exp[
                filter_word][
                alg_name][
                is_ast][
                is_pre
            ].get(_data):
                exp[
                    filter_word][
                    alg_name][
                    is_ast][
                    is_pre][
                    _data
                ] = {}

            _num_clu = row[1]
            _score = row[3]

            if not exp[
                filter_word][
                alg_name][
                is_ast][
                is_pre][
                _data
            ].get(_num_clu):
                exp[
                    filter_word][
                    alg_name][
                    is_ast][
                    is_pre][
                    _data][
                    _num_clu
                ] = _score

    # 1 - data (filter word)
    #   2 - algorithm
    #       3 - is ast?
    #           4 - preprocess?
    #               5 - vector space
    #                   6 - cluster num

    # TODO: fix plotting creating, add auto
    plt.rcParams["figure.figsize"] = (15, 10)
    for _filter in exp.keys():
        # prepare axes
        algs = sorted(exp[_filter].keys())
        fig, axs = plt.subplots(2, 3)
        fig.suptitle(
            f'Эксперимент со {"всем набором данных" if _filter == "_" else "словами-фильтрами: "+_filter}'
        )

        # 1 - tf-idf chars
        axs[0, 0].cla()
        axs[0, 0].grid()
        axs[0, 0].set_title('а) TF-IDF все символы')
        axs[0, 0].set_xlabel('Кол-во кластеров')
        axs[0, 0].set_ylabel('Оценка силуэта')
        # 2 - tf-idf non-terminals
        axs[0, 1].cla()
        axs[0, 1].grid()
        axs[0, 1].set_title('б) TF-IDF не терминалы')
        axs[0, 1].set_xlabel('Кол-во кластеров')
        axs[0, 1].set_ylabel('Оценка силуэта')
        # 3 - tf-idf tokens
        axs[0, 2].cla()
        axs[0, 2].grid()
        axs[0, 2].set_title('в) TF-IDF токены')
        axs[0, 2].set_xlabel('Кол-во кластеров')
        axs[0, 2].set_ylabel('Оценка силуэта')
        # 4 - bert
        axs[1, 0].cla()
        axs[1, 0].grid()
        axs[1, 0].set_title('г) BERT')
        axs[1, 0].set_xlabel('Кол-во кластеров')
        axs[1, 0].set_ylabel('Оценка силуэта')
        # 5 - code bert
        axs[1, 1].cla()
        axs[1, 1].grid()
        axs[1, 1].set_title('д) CodeBERT')
        axs[1, 1].set_xlabel('Кол-во кластеров')
        axs[1, 1].set_ylabel('Оценка силуэта')
        # 6 - modern bert
        axs[1, 2].cla()
        axs[1, 2].grid()
        axs[1, 2].set_title('е) ModernBERT')
        axs[1, 2].set_xlabel('Кол-во кластеров')
        axs[1, 2].set_ylabel('Оценка силуэта')

        for _alg in algs:
            for _ast in exp[_filter][_alg].keys():
                for _pre in exp[_filter][_alg][_ast].keys():
                    for _space in exp[_filter][_alg][_ast][_pre].keys():

                        data = exp[_filter][_alg][_ast][_pre][_space].items()
                        x = sorted([int(x[0]) for x in data])
                        y = [x[1] for x in data]

                        if 'tf_idf_chars' in _alg:
                            i, j = 0, 0
                        elif 'tf_idf_non_terminals' in _alg:
                            i, j = 0, 1
                        elif 'tf_idf_tokens' in _alg:
                            i, j = 0, 2
                        elif 'bert_base_uncased' in _alg:
                            i, j = 1, 0
                        elif 'bert_base_code' in _alg:
                            i, j = 1, 1
                        elif 'bert_base_modern' in _alg:
                            i, j = 1, 2
                        else:
                            print(f'Error with <{_alg}>')
                            raise NotImplementedError

                        _label = f'{"АСД" if _ast == "ast" else "РВ"} ' \
                                 f'| {"С пред." if _pre == "pre" else "Без пред."} ' \
                                 f'| {"2-е пр-во" if _space == "_data_2d" else "Исх. пр-во"}'

                        axs[i, j].plot(
                            x, y,
                            label=_label,
                            linestyle=random.choice(line_styles)
                        )

        axs[0, 0].legend(loc="best")
        axs[0, 1].legend(loc="best")
        axs[0, 2].legend(loc="best")
        axs[1, 0].legend(loc="best")
        axs[1, 1].legend(loc="best")
        axs[1, 2].legend(loc="best")

        plt.tight_layout()
        plt.savefig(
            Path(
                'clustering_results_for_report',
                f'{"all" if _filter == "_" else _filter}.png'
            )
        )
