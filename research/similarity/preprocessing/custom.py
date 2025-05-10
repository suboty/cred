import os
import re
import unicodedata
from ast import literal_eval
from typing import Tuple, Optional, List

import networkx

from utils import Stack


MAX_LEN = 'inf'


class LexicalAnalyzerError(Exception):
    ...


class LexicalAnalyzer:
    MIN_CHAR = 0x20
    MAX_CHAR = 0x100000
    PROGRESS = 0x8000

    s2t = {
        # characters
        'alt': '|',
        'any': '.',

        # quantifiers
        '0_or_1': '?',
        '1_or_more': '+',
        '0_or_more': '*',

        # bracket groups
        'start_range': '[',
        'end_range': ']',
        'start_group': '(',
        'end_group': ')',
        'start_quant': '{',
        'end_quant': '}',

        # special
        'escape': '\\'
    }

    f2t = {

    }

    brackets = ['(', '[', '{', ')', ']', '}']
    bracket_pairs = {
        ')': '(',
        ']': '[',
        '}': '{',
    }
    bracket_ids = {
        '(': -1,
        '[': -1,
        '{': -1
    }

    # TODO: add groups meta like [^], (?!), ...
    groups_meta = {
        '(': ['|']
    }

    def __init__(self):
        self.tokens = []
        self.brackets_stack = Stack()

        self._escape = False
        if not os.path.isfile('unicode'):
            self.make_unicode_characters_storage()

        self.atoms = self.get_unicode_characters()
        super().__init__()
        self.__dialect_id = 1

    def make_unicode_characters_storage(self):
        with open('unicode-chars', 'w') as file:
            for i in range(self.MIN_CHAR, self.MAX_CHAR):
                char = chr(i)
                try:
                    name = unicodedata.name(char)
                    codepoint = hex(i)[2:].rjust(5, 'w').upper()
                    file.write("%s\t%s\t%s\n" % (codepoint, char, name))
                except ValueError as e:
                    pass

    @staticmethod
    def get_unicode_characters():
        with open('unicode-chars', 'r') as file:
            chars = file.readlines()
        return [x for char in chars for x in char.split('\t')[1]]

    def __call__(
            self,
            regex: str) -> List:
        self.tokens = []
        regex = '(' + regex + ')'
        for symbol in regex:
            self.eat(symbol)
        return self.tokens

    def eat(self, symbol):
        _find = False

        if self._escape:
            self.tokens.append(f'escape,{symbol}')
            _find = True
            self._escape = False

        if not _find:
            for key in self.f2t.keys():
                if symbol == self.s2t.get(key):
                    self.tokens.append(f'{key}')
                    _find = True

        if not _find:
            for key in self.s2t.keys():
                if symbol == self.s2t.get(key):

                    if key == 'escape':
                        self._escape = True
                        _find = True
                    elif key == 'alt':
                        self.tokens.append(f'{key},{self.bracket_ids["("]}')
                        _find = True
                    elif symbol in self.brackets:
                        if not self.bracket_pairs.get(symbol):
                            self.brackets_stack.push(symbol)
                            self.bracket_ids[symbol] += 1
                            self.tokens.append(f'{key},{self.bracket_ids[symbol]}')
                            _find = True
                        else:
                            if self.bracket_pairs.get(symbol) == self.brackets_stack.get(-1):
                                __current_bracket = self.bracket_pairs[symbol]
                                self.brackets_stack.pop()
                                self.tokens.append(f'{key},{self.bracket_ids[__current_bracket]}')
                                _find = True
                                self.bracket_ids[__current_bracket] -= 1
                            elif self.brackets_stack.get(-1) == 0:
                                __current_bracket = self.bracket_pairs[symbol]
                                self.tokens.append(f'{key},{self.bracket_ids[__current_bracket]}')
                                _find = True
                                self.bracket_ids[__current_bracket] -= 1
                            else:
                                raise LexicalAnalyzerError('Parenthesis mismatch')
                    else:
                        self.tokens.append(f'{key}')
                        _find = True
        if not _find:
            self.tokens.append(f'atom,{symbol}')


class SyntaxAnalyzer:
    state = None
    current_string = ''
    history = ''

    range_reg = re.compile(r'\d\-\d|\w\-\w')

    def set_state(self, state):
        self.state = state
        self.history += f'->{state}'

    @staticmethod
    def atom(tree: List, token: str, value: Optional[str]):
        match token:
            case 'any':
                tree.append(('any',))
            case 'escape':
                tree.append(('escape', value))
            case 'atom':
                tree.append(('atom', value))
            case '0_or_1' | '1_or_more' | '0_or_more':
                previous_token = tree[-1]
                tree[-1] = (
                    (
                        f'repeat_{token.replace("_or_", "_").replace("more", MAX_LEN)}',
                        previous_token
                    )
                )

        return tree

    @staticmethod
    def eat(current_string):

        def get_token_meta(token):
            try:
                token_name, value = token.split(',', 1)
            except ValueError:
                token_name = token
                value = None
            return token_name, value

        try:
            current_token = current_string[0]
            current_string = current_string[1:]
            return current_string, get_token_meta(current_token)
        except IndexError:
            return None, None

    @staticmethod
    def tokens_split(tokens, split_token):
        parts = []
        _part = []
        for token in tokens:
            if token == split_token:
                parts.append(_part)
                _part = []
            else:
                _part.append(token)
        parts.append(_part)
        return parts

    # TODO: add validate for syntax constructions in ranges and quantifiers
    def group(self, current_string: List, tree: List, token: str, value=str):
        match token:
            case 'start_group':
                parts = []
                _alt = None
                _current_token = 'start_group'
                _current_value = value
                tokens = []
                while True:
                    current_string, _token_meta = self.eat(current_string)
                    if not current_string:
                        break
                    _current_token, _value = [x for x in _token_meta]
                    if _value == _current_value and _current_token == 'end_group':
                        break
                    # TODO: add validate id of alt token
                    if _value == _current_value and _current_token == 'alt':
                        _alt = f'{_current_token},{_value if _value else ""}'
                    tokens.append(f'{_current_token},{_value if _value else ""}')
                if _alt:
                    parts = self.tokens_split(
                        tokens=tokens,
                        split_token=_alt
                    )
                if parts:
                    trees = []
                    for part in parts:
                        trees.append((
                            'altgroup',
                            tuple(self.get_tree(part))
                        ))
                    tree.append(
                        (
                            'group',
                            (
                                'alt',
                                tuple(trees)
                            ),
                        )
                    )
                else:
                    tree.append(
                        (
                            'group',
                            tuple(self.get_tree(tokens))
                        )
                    )
            case 'start_range':
                # TODO: add special constructions like [^]
                _current_token = 'start_range'
                _current_value = value
                _params = ''
                while True:
                    current_string, _token_meta = self.eat(current_string)
                    _current_token, _value = [x for x in _token_meta]
                    if _value == _current_value and _current_token == 'end_range':
                        break
                    _params += _value
                _params = self.range_reg.findall(_params)
                for _param in _params:
                    range_name = 'range_' + _param.replace('-', '_')
                    tree.append(
                        (
                            range_name,
                        )
                    )
            case 'start_quant':
                previous_token = tree[-1]
                _current_token = 'start_quant'
                _current_value = value
                _params = ''
                while True:
                    current_string, _token_meta = self.eat(current_string)
                    _current_token, _value = [x for x in _token_meta]
                    if _value == _current_value and _current_token == 'end_quant':
                        break
                    _params += _value
                _len_params = len(_params)
                repeat_name = 'repeat_'
                if _len_params == 1:
                    # case like a{1}
                    repeat_name += f'{_params[0]}'
                elif _len_params == 3:
                    # case like a{1,2}
                    repeat_name += '_'.join(_params.split(','))
                elif _len_params == 2:
                    # case like a{0,}
                    if _params[0] != ',':
                        repeat_name += f'{_params[0]}_more'
                    else:
                        raise AttributeError
                else:
                    raise AttributeError
                tree[-1] = (
                    (
                        repeat_name.replace('more', MAX_LEN),
                        previous_token
                    )
                )
        return current_string, tree

    @staticmethod
    def end(tree):
        tuple_tree = tuple(tree[0])
        return 'seq', tuple_tree[1]

    def get_tree(self, current_string):
        tree = []
        while current_string:
            current_string, token_meta = self.eat(current_string)
            token_name, value = [x for x in token_meta]
            match token_name:
                case 'any':
                    self.set_state('any')
                    tree = self.atom(
                        tree=tree,
                        token=token_name,
                        value=value,
                    )
                case '0_or_1' | '1_or_more' | '0_or_more':
                    self.set_state('quantifier')
                    tree = self.atom(
                        tree=tree,
                        token=token_name,
                        value=value,
                    )
                case 'start_group' | 'start_range' | 'start_quant':
                    self.set_state(token_name.replace('start_', ''))
                    current_string, tree = self.group(
                        current_string=current_string,
                        tree=tree,
                        token=token_name,
                        value=value,
                    )
                case _:
                    self.set_state('atom')
                    tree = self.atom(
                        tree=tree,
                        token=token_name,
                        value=value,
                    )
        return tree

    def __call__(self, tokens: List, *args, **kwargs):
        self.current_string = tokens
        result = self.get_tree(current_string=self.current_string)
        return self.end(result)


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

    def translate(self, regex):
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
            if res is None:
                words_dict[node] = 0
                return 0
            else:
                words_dict[node] += 1
                return words_dict[node]

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

    def __call__(self, regex):
        ast = self.translate(regex)
        ast = self.postprocess(ast)
        ast_graph = self.get_graph(ast)
        return ast_graph


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    translator = CustomTranslator(verbose=True)
    test_regex = input('Input regex: ')

    test_ast = translator.translate(test_regex)
    print(f'AST: {test_ast}')
    processed_ast = translator.postprocess(test_ast)
    print(f'Post AST: {processed_ast}')

    test_ast_graph = translator.get_graph(processed_ast)
    networkx.draw_networkx(test_ast_graph)
    plt.show()
