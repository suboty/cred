import re
from ast import literal_eval
from typing import Tuple, Optional

import networkx

from research.similarity.preprocessing.custom_translator.syntax_analyzer import SyntaxAnalyzer
from research.similarity.preprocessing.custom_translator.lexical_analyzer import LexicalAnalyzer


class CustomTranslator:
    def __init__(self, verbose: bool = False):
        self.errors = 0
        self.verbose = verbose

        self.la = LexicalAnalyzer()
        self.sa = SyntaxAnalyzer()

        self.clean_sub_regexes = [
            # atom cleaning
            (re.compile(r"atom\'\,\s[^)]*"), "atom',"),
            # escape cleaning
            (re.compile(r"escape\'\,\s[^)]*"), "escape',"),
        ]

    def __repr__(self):
        return 'custom_translator'

    def __call__(self, regex):
        tokens = self.la(regex)
        ast = self.sa(tokens)
        return ast

    def postprocess(self, ast):
        str_ast = str(ast)
        for reg in self.clean_sub_regexes:
            str_ast = reg[0].sub(
                string=str_ast,
                repl=reg[1],
            )
        return literal_eval(str_ast)

    @staticmethod
    def get_graph(_ast):
        G = networkx.Graph()
        G.add_node('seq')

        words_dict = {}

        def get_id(node):
            res = words_dict.get(node)
            if res is not None:
                words_dict[node] += 1
                return res
            else:
                words_dict[node] = 0
                return 0

        def parse(tree, graph, root: Optional[str] = None):
            if not root:
                root = 'seq'
            if isinstance(tree, Tuple):
                for subtree in tree:
                    root, graph = parse(
                        tree=subtree,
                        graph=graph,
                        root=root
                    )
            elif isinstance(tree, str):
                _node = f'{tree},{get_id(tree)}'
                graph.add_node(_node)
                graph.add_edge(root, _node)
                root = _node
            return root, graph

        # seq node skip
        _ast = _ast[1]
        _, g = parse(tree=_ast, graph=G)

        return g


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    translator = CustomTranslator(verbose=True)
    test_regex = input('Input regex: ')

    test_ast = translator(test_regex)
    print(f'AST: {test_ast}')
    processed_ast = translator.postprocess(test_ast)
    print(f'Post AST: {processed_ast}')

    test_ast_graph = translator.get_graph(processed_ast)
    networkx.draw_networkx(test_ast_graph)
    plt.show()
