import os
import unicodedata
from typing import List

from research.similarity.preprocessing.custom_translator import Stack


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
