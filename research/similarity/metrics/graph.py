from typing import Callable, Optional

import networkx as nx

from research.similarity.preprocessing.sre import SreParser


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
        kwargs.setdefault('roots', ('EXPR', 'EXPR'))

        # optional params
        if node_match:
            kwargs.setdefault('node_match', node_match)
        if edge_match:
            kwargs.setdefault('edge_match', edge_match)

        if is_optimize:
            return nx.optimize_graph_edit_distance(
                graph1,
                graph2,
                **kwargs
            )

        return nx.graph_edit_distance(
            graph1,
            graph2,
            **kwargs
        )


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import networkx as nx

    test_parser = SreParser()
    regex1 = input('Input regex #1: ')
    regex2 = input('Input regex #2: ')

    test_graph1 = test_parser(regex1)
    test_graph2 = test_parser(regex2)

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)

    nx.draw_networkx(
        test_graph1,
        ax=ax1,
        font_size=6,
        node_color='#aed6f1'
    )
    ax1.margins(0.3)
    ax1.set_title(f'Regex1\n{regex1}')
    ax1.axis('off')

    nx.draw_networkx(
        test_graph2,
        ax=ax2,
        font_size=6,
        node_color='#aed6f1'
    )
    ax2.margins(0.3)
    ax2.set_title(f'Regex2\n{regex2}')
    ax2.axis('off')

    ged = GraphSimilarity.get_graph_edit_distance(
        test_graph1,
        test_graph2
    )
    f.suptitle(f'Regexes GED is {ged}')

    plt.show()
