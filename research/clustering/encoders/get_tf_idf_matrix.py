import json
from pathlib import Path
from collections import Counter
from typing import Dict, Union, List, Tuple

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import get_all_unicode_letters


def load_tokens_rules(
        path_to_1_symbol_rules: Union[str, Path],
        path_to_2_or_more_symbols_rules: Union[str, Path],
) -> Dict:
    """Load token rules for lexical analyzing into Python dict."""

    def _process_rules(_rules: List) -> List[Tuple]:
        """Split every rule to tuple (symbols, token_name)."""
        new_rules = []
        for rule in _rules:
            # remove newline from end of rule
            if rule[-1:] == '\n':
                rule = rule[:-1]
            # add escape symbol
            rule = rule.replace('<esc>', '\\')
            rule = rule.split(' -> ')
            new_rules.append(tuple(rule))
        return new_rules

    rules = {}
    with open(path_to_1_symbol_rules, 'r') as file_1_symbol:
        rules['1'] = {
            x[0]: x[1] for x in
            _process_rules(file_1_symbol.readlines())
        }
    with open(path_to_2_or_more_symbols_rules, 'r') as file_2_or_more_symbol:
        rules_2_or_more = _process_rules(file_2_or_more_symbol.readlines())
        # sort by length of symbols length
        rules_2_or_more = sorted(
            rules_2_or_more,
            key=lambda x: len(x[0]),
            reverse=True
        )
        rules['2_or_more'] = {x[0]: x[1] for x in rules_2_or_more}
    return rules


TOKENS_RULES = load_tokens_rules(
    path_to_1_symbol_rules=Path('encoders', 'tokens_rules_1_symbol'),
    path_to_2_or_more_symbols_rules=Path('encoders', 'tokens_rules_2_or_more_symbols'),
)

TOKENS_SEP = 'SEP-SYMBOL'
IS_NEED_SCALING = False


class TfidfMatrix:

    scaler = MinMaxScaler()

    @staticmethod
    def scaling_decorator(func):
        def wrapper(*args, **kwargs):
            encoder, csr_matrix = func(*args, **kwargs)
            csr_matrix = csr_matrix.toarray()
            if IS_NEED_SCALING:
                csr_matrix = TfidfMatrix.scaler.fit_transform(csr_matrix)
            return encoder, csr_matrix

        return wrapper

    @staticmethod
    @scaling_decorator
    def get_matrix_tokenize_by_chars(
            list_of_regexes,
    ):
        """Get TF-IDF matrix by chars."""
        chars_tfidf_encoder = TfidfVectorizer(analyzer='char')
        chars_tfidf_matrix = chars_tfidf_encoder.fit_transform(list_of_regexes)
        return chars_tfidf_encoder, chars_tfidf_matrix

    @staticmethod
    @scaling_decorator
    def get_matrix_tokenize_by_non_terminals(
            list_of_regexes,
    ):
        """Get TF-IDF matrix by non-terminals."""

        # select special characters, which used in regular expressions as a non-terminal symbols
        special_chars = []
        # from "space" to "slash"
        special_chars += get_all_unicode_letters('0020', '002F')
        # from "colon" to "at symbol"
        special_chars += get_all_unicode_letters('003A', '0040')
        # from "open square bracket" to "underscore"
        special_chars += get_all_unicode_letters('005B', '005F')
        # from "open curly bracket" to "tilda"
        special_chars += get_all_unicode_letters('007B', '007E')

        def custom_tokenize(text):
            return [x for x in list(text) if x in special_chars]

        non_terminals_tfidf_encoder = TfidfVectorizer(
            tokenizer=custom_tokenize,
            analyzer='word'
        )
        non_terminals_tfidf_matrix = non_terminals_tfidf_encoder.fit_transform(list_of_regexes)
        return non_terminals_tfidf_encoder, non_terminals_tfidf_matrix

    @staticmethod
    @scaling_decorator
    def get_matrix_tokenize_by_regex_tokens(
            list_of_regexes,
    ):
        """Get TF-IDF matrix by custom regex tokens."""

        tokens_stats = []

        def custom_tokenize(text):
            tokens = []
            # work with 2 or more symbols tokens
            rules_2_or_more_symbol_tokens_keys = TOKENS_RULES['2_or_more'].keys()
            new_text = text
            for token in rules_2_or_more_symbol_tokens_keys:
                new_text = new_text.replace(
                    token,
                    TOKENS_SEP * 2 + TOKENS_RULES['2_or_more'].get(token) + TOKENS_SEP * 2,
                )
            new_text = new_text.split(TOKENS_SEP)

            if new_text[-1] == '':
                new_text = new_text[:-1]

            if new_text[0] == '':
                new_text = new_text[1:]

            # work with 1 symbol tokens
            rules_1_symbol_tokens_keys = TOKENS_RULES['1'].keys()

            is_2_token = False
            for part in new_text:

                # flag
                if part == '':
                    if is_2_token:
                        is_2_token = False
                    else:
                        is_2_token = True
                    continue

                if is_2_token:
                    tokens.append(part)
                else:
                    for symbol in part:
                        if symbol in rules_1_symbol_tokens_keys:
                            tokens.append(TOKENS_RULES['1'].get(symbol))
                            continue
                        tokens.append(f'atom, {symbol}')
            tokens_stats.append([x for x in tokens if 'atom, ' not in x])
            return tokens

        tokens_tfidf_encoder = TfidfVectorizer(
            tokenizer=custom_tokenize,
            analyzer='word'
        )
        tokens_tfidf_matrix = tokens_tfidf_encoder.fit_transform(list_of_regexes)

        # save tokens statistics
        _stats = []
        for stat in tokens_stats:
            _stats += stat
        tokens_stats = dict(Counter(_stats).most_common(20))
        with open(Path('tmp', 'tokens_stats.json'), 'w') as f:
            json.dump(tokens_stats, f, ensure_ascii=False)

        return tokens_tfidf_encoder, tokens_tfidf_matrix
