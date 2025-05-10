import re
try:
    import re._parser as sre_parse
except ImportError:
    # Python < 3.11
    from re import sre_parse


from typing import List, Union, Tuple, Optional
from ast import literal_eval

import networkx

from logger import logger


class SreParser:
    def __init__(self, verbose: bool = False):
        self.errors = 0
        self.verbose = verbose

        self.reg_clean = re.compile(r'\(LITERAL\,\s\d*\)')

    def __repr__(self):
        return 'sre_parser'

    @staticmethod
    def get_graph(ast):
        words_dict = {}

        # prepare ast for graph parsing
        ast = ast.replace('_', '')
        words = re.findall(
            pattern=r'[^\'a-zA-Z]([a-zA-Z]+)[^\'a-zA-Z]',
            string=ast
        )
        words = sorted(list(set(words)), reverse=True)
        for word in words:
            words_dict[word] = 0
            ast = re.sub(
                pattern=f'(?<!(\w))({word})',
                string=ast,
                repl=f"'{word}'"
            )
        ast = literal_eval(ast)

        # parse for graph (NetworkX object)
        G = networkx.Graph()
        G.add_node('EXPR')

        def get_id(node):
            res = words_dict.get(node)
            if res is not None:
                words_dict[node] += 1
                return res
            else:
                words_dict[node] = 0
                return 0

        ###
        def sre_ast_to_graph(_ast, graph, root: Optional[str] = None):
            if not root:
                root = 'EXPR'
            if isinstance(_ast, Union[List, Tuple]):
                for subtree in _ast:
                    root, graph = sre_ast_to_graph(
                        _ast=subtree,
                        graph=graph,
                        root=root
                    )
            elif isinstance(_ast, Union[str, int, None]):
                _node = f'{_ast},{get_id(_ast)}'
                graph.add_node(_node)
                graph.add_edge(root, _node)
                root = _node
            return root, graph

        return sre_ast_to_graph(_ast=ast, graph=G)
        ###

    def parse(self, regex):
        try:
            return sre_parse.parse(regex)
        except Exception as e:
            if self.verbose:
                logger.warning(
                    f'This expression <{regex}> does not '
                    f'written by python flavor: {e}'
                )
            self.errors += 1
            return None

    def postprocess(self, ast):
        try:
            clean_ast = self.reg_clean.sub(
                string=str(ast),
                repl='LITERAL'
            )
            res_ast = clean_ast
        except Exception as e:
            if self.verbose:
                logger.warning(
                    f'Error while this ast <{ast}> '
                    f'cleaning: {e}'
                )
            res_ast = ast
        return res_ast

    def parse_list(
            self,
            regex_list: List,
            is_need_postprocessing: bool = True,
            is_need_graph_representation: bool = True
    ):
        ast = []
        for i, regex in enumerate(regex_list):
            parsed_regex = self.parse(regex)
            if parsed_regex:
                if is_need_postprocessing:
                    ast.append(
                        self.postprocess(
                            parsed_regex
                        )
                    )
                else:
                    ast.append(
                        str(parsed_regex)
                    )
        if self.errors > 0:
            logger.warning(
                f'{self.errors} regexes does not '
                f'written by python flavor'
            )
        self.errors = 0
        if is_need_graph_representation:
            return [self.get_graph(x) for x in ast]

    def __call__(self, regex):
        ast = self.parse(regex)
        ast = self.postprocess(ast)
        _, ast_graph = self.get_graph(ast)
        return ast_graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    parser = SreParser(verbose=True)
    test_regex = input('Input regex: ')

    test_ast = parser.parse(test_regex)
    print(f'AST: {test_ast}')
    processed_ast = parser.postprocess(test_ast)
    print(f'Post AST: {processed_ast}')

    _, test_ast_graph = parser.get_graph(processed_ast)
    networkx.draw_networkx(test_ast_graph)
    plt.show()
