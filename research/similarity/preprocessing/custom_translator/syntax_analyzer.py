import re
from typing import List, Optional


MAX_LEN = 'inf'


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
            # TODO: add dialect id validate
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
