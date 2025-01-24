from pathlib import Path
from typing import Dict, Union, List, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer


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


class TfidfMatrix:
    @staticmethod
    def get_matrix_tokenize_by_chars(
            list_of_regexes,
    ):
        """Get TF-IDF matrix by chars."""
        chars_tfidf_encoder = TfidfVectorizer(analyzer='char')
        chars_tfidf_matrix = chars_tfidf_encoder.fit_transform(list_of_regexes)
        return chars_tfidf_encoder, chars_tfidf_matrix

    @staticmethod
    def get_matrix_tokenize_by_non_terminals(
            list_of_regexes,
            special_chars,
    ):
        """Get TF-IDF matrix by non-terminals."""

        def custom_tokenize(text):
            return [x for x in list(text) if x in special_chars]

        non_terminals_tfidf_encoder = TfidfVectorizer(
            tokenizer=custom_tokenize,
            analyzer='word'
        )
        non_terminals_tfidf_matrix = non_terminals_tfidf_encoder.fit_transform(list_of_regexes)
        return non_terminals_tfidf_encoder, non_terminals_tfidf_matrix

    @staticmethod
    def get_matrix_tokenize_by_regex_tokens(
            list_of_regexes,
    ):
        """Get TF-IDF matrix by custom regex tokens."""
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

            return tokens

        tokens_tfidf_encoder = TfidfVectorizer(
            tokenizer=custom_tokenize,
            analyzer='word'
        )
        tokens_tfidf_matrix = tokens_tfidf_encoder.fit_transform(list_of_regexes)
        return tokens_tfidf_encoder, tokens_tfidf_matrix
