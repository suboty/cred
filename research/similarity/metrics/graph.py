from typing import Callable, Optional

import networkx as nx


class GraphSimilarity:
    """
    Class for graph similarity calculating.
    Build on NetworkX library.
    See in https://networkx.org/documentation/stable/reference/algorithms/similarity.html.
    """

    @staticmethod
    def get_graph_edit_distance(
            graph1: nx.Graph,
            graph2: nx.Graph,
            node_match: Optional[Callable] = None,
            edge_match: Optional[Callable] = None,
            is_optimize: bool = False,
    ):
        """
        Graph Edit Distance (GED).

        Graph edit distance is a graph similarity measure
        analogous to Levenshtein distance for strings.
        It is defined as minimum cost of edit path
        transforming graph1
        to graph isomorphic to graph2.

        :param graph1: First graph for comparison.
        :param graph2: Second graph for comparison.
        :param node_match: A function that returns True
            if node n1 in G1 and n2 in G2 should be
            considered equal during matching.

            The function will be called like:
            node_match(G1.nodes[n1], G2.nodes[n2])
        :param edge_match: A function that returns True
            if the edge attribute dictionaries
            for the pair of nodes (u1, v1) in G1
            and (u2, v2) in G2 should be
            considered equal during matching.

            The function will be called like:
            edge_match(G1[u1][v1], G2[u2][v2])
        :param is_optimize: Need optimized calculations?
        :return: GED (float)
        """
        kwargs = {}

        if not is_optimize:
            kwargs.setdefault('roots', ('EXPR', 'EXPR'))

        # optional params
        if node_match:
            kwargs.setdefault('node_match', node_match)
        if edge_match:
            kwargs.setdefault('edge_match', edge_match)

        if is_optimize:
            minv = None
            for v in nx.optimize_graph_edit_distance(
                graph1,
                graph2,
                **kwargs
            ):
                minv = v
            return minv

        return nx.graph_edit_distance(
            graph1,
            graph2,
            **kwargs
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx

    from research.similarity.preprocessing.sre import SreParser
    from research.similarity.preprocessing.custom import CustomTranslator

    test_parser = SreParser()
    test_translator = CustomTranslator()

    # regex1 = input('Input regex #1: ')
    # regex2 = input('Input regex #2: ')

    regex1 = 'A|B'
    regex2 = 'C|VC'

    test_graph11 = test_parser(regex1)
    test_graph12 = test_parser(regex2)

    ged_sre = GraphSimilarity.get_graph_edit_distance(
        test_graph11, test_graph12,
        is_optimize=True
    )

    test_graph21 = test_translator(regex1)
    test_graph22 = test_translator(regex2)

    ged_custom = GraphSimilarity.get_graph_edit_distance(
        test_graph21, test_graph22,
        is_optimize=True
    )

    f, axs = plt.subplots(2, 3, sharey=True)
    f.set_size_inches(10, 8, forward=True)
    f.subplots_adjust(top=0.8)

    def ax_plot(_graph, _ax, _regex, _regex_id):
        nx.draw_networkx(
            G=_graph,
            ax=_ax,
            font_size=6,
            node_color='#aed6f1'
        )
        _ax.margins(0.3)
        _ax.set_title(f'Regex{_regex_id}\n{_regex}')
        _ax.axis('off')

    # for SRE parser
    ax_plot(
        _graph=test_graph11,
        _ax=axs[0, 0],
        _regex=regex1,
        _regex_id=1
    )
    ax_plot(
        _graph=test_graph12,
        _ax=axs[0, 1],
        _regex=regex2,
        _regex_id=2
    )
    axs[0, 2].text(0.5, 0.5, f'SRE Parser\nRegexes GED: {ged_sre}')
    axs[0, 2].axis('off')

    # for custom translator
    ax_plot(
        _graph=test_graph21,
        _ax=axs[1, 0],
        _regex=regex1,
        _regex_id=1
    )
    ax_plot(
        _graph=test_graph22,
        _ax=axs[1, 1],
        _regex=regex2,
        _regex_id=2
    )
    axs[1, 2].text(0.5, 0.5, f'Custom Translator\nRegexes GED: {ged_custom}')
    axs[1, 2].axis('off')

    f.suptitle(f'Regexes Similarity')

    plt.tight_layout()
    plt.show()
