import re
import sre_parse
from collections import defaultdict
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path


LIMIT = 65535
ALPHABET = []


@dataclass
class PatternInfo:
    pattern: str # input pattern
    replaced: str # replaced pattern
    category: str # category for pattern


def get_all_unicode_letters(start_code, stop_code):
    start_idx, stop_idx = [int(code, 16) for code in (start_code, stop_code)]
    characters = []
    for unicode_idx in range(start_idx, stop_idx + 1):
        characters.append(chr(unicode_idx))
    return characters


# Latin
ALPHABET += get_all_unicode_letters('0041', '005A')  # A-Z
ALPHABET += get_all_unicode_letters('0061', '007A')  # a-z
ALPHABET += get_all_unicode_letters('0030', '0039')  # 0-9

# Cyrillic
ALPHABET += get_all_unicode_letters('0400', '045F')  # Ѐ-џ
ALPHABET += get_all_unicode_letters('0460', '047F')  # Ѡ-ѿ
ALPHABET += get_all_unicode_letters('0480', '04FF')  # Ҁ-ӿ

# Greek
ALPHABET += get_all_unicode_letters('0370', '03FF')  # Ͱ-Ͽ

# Arabic
ALPHABET += get_all_unicode_letters('0600', '06FF')  # Arabic
ALPHABET += get_all_unicode_letters('0750', '077F')  # Arabic Supplement

# Hebrew
ALPHABET += get_all_unicode_letters('0590', '05FF')  # Hebrew

# Chinese/Japanese/Korean (CJK)
ALPHABET += get_all_unicode_letters('4E00', '9FFF')  # CJK Unified Ideographs
ALPHABET += get_all_unicode_letters('3400', '4DBF')  # CJK Unified Ideographs Extension A
ALPHABET += get_all_unicode_letters('F900', 'FAFF')  # CJK Compatibility Ideographs

# Hiragana & Katakana (Japanese)
ALPHABET += get_all_unicode_letters('3040', '309F')  # Hiragana
ALPHABET += get_all_unicode_letters('30A0', '30FF')  # Katakana

# Hangul (Korean)
ALPHABET += get_all_unicode_letters('AC00', 'D7AF')  # Hangul Syllables
ALPHABET += get_all_unicode_letters('1100', '11FF')  # Hangul Jamo

# Devanagari (Hindi, Sanskrit)
ALPHABET += get_all_unicode_letters('0900', '097F')  # Devanagari

# Bengali
ALPHABET += get_all_unicode_letters('0980', '09FF')  # Bengali

# Thai
ALPHABET += get_all_unicode_letters('0E00', '0E7F')  # Thai

# Georgian
ALPHABET += get_all_unicode_letters('10A0', '10FF')  # Georgian

# Armenian
ALPHABET += get_all_unicode_letters('0530', '058F')  # Armenian

# Extended Latin
ALPHABET += get_all_unicode_letters('00C0', '00FF')  # Latin-1 Supplement
ALPHABET += get_all_unicode_letters('0100', '017F')  # Latin Extended-A
ALPHABET += get_all_unicode_letters('0180', '024F')  # Latin Extended-B
ALPHABET += get_all_unicode_letters('1E00', '1EFF')  # Latin Extended Additional


def ast_to_literal_string(ast) -> str:
    result = []

    for op, value in ast:
        if op == sre_parse.LITERAL:
            char = chr(value)
            if char in '.^$*+?{}[]()|\\':
                result.append(f'\\{char}')
            else:
                result.append(char)
        elif op == sre_parse.RANGE:
            start, end = value
            result.append(f'{chr(start)}-{chr(end)}')
        elif op == sre_parse.CATEGORY:
            if value == sre_parse.CATEGORY_DIGIT:
                result.append(r'\d')
            elif value == sre_parse.CATEGORY_WORD:
                result.append(r'\w')
            elif value == sre_parse.CATEGORY_SPACE:
                result.append(r'\s')
            elif value == sre_parse.CATEGORY_NON_DIGIT:
                result.append(r'\D')
            elif value == sre_parse.CATEGORY_NON_WORD:
                result.append(r'\W')
            elif value == sre_parse.CATEGORY_NON_SPACE:
                result.append(r'\S')
            else:
                return ''
        elif op == sre_parse.IN:
            inner = ast_to_literal_string(value)
            if not inner:
                return ''
            result.append('[')
            result.append(inner)
            result.append(']')
        elif op == sre_parse.MAX_REPEAT:
            min_count, max_count, subpattern = value
            inner = ast_to_literal_string(subpattern)
            if not inner:
                return ''
            if min_count == 0 and max_count == 1:
                result.append(f'{inner}?')
            elif min_count == 0 and max_count == LIMIT:
                result.append(f'{inner}*')
            elif min_count == 1 and max_count == LIMIT:
                result.append(f'{inner}+')
            elif min_count == max_count:
                result.append(f'{inner}{{{min_count}}}')
            else:
                result.append(f'{inner}{{{min_count},{max_count}}}')
        elif op == sre_parse.MIN_REPEAT:
            min_count, max_count, subpattern = value
            inner = ast_to_literal_string(subpattern)
            if not inner:
                return ''
            result.append(f'{inner}{{{min_count},{max_count}}}?')
        elif op == sre_parse.BRANCH:
            _, branches = value
            branch_strings = []
            for branch in branches:
                branch_str = ast_to_literal_string(branch)
                if not branch_str:
                    return ''
                branch_strings.append(branch_str)
            result.append('(')
            result.append('|'.join(branch_strings))
            result.append(')')
        elif op == sre_parse.SUBPATTERN:
            group_id, flags, subpattern = value
            inner = ast_to_literal_string(subpattern)
            if not inner:
                return ''
            if flags:
                flags_str = ''
                if flags & re.IGNORECASE:
                    flags_str += 'i'
                if flags & re.MULTILINE:
                    flags_str += 'm'
                if flags & re.DOTALL:
                    flags_str += 's'
                result.append(f'(?{flags_str}:{inner})')
            elif group_id:
                result.append(f'(?P<{group_id}>{inner})')
            else:
                result.append(f'({inner})')
        else:
            return ''

    return ''.join(result)


def extract_char_class(class_ast, patterns: List[PatternInfo]):
    is_negated = False
    if class_ast and class_ast[0][0] == sre_parse.NEGATE:
        is_negated = True
        class_ast = class_ast[1:]

    if len(class_ast) == 1:
        item = class_ast[0]

        if item[0] == sre_parse.RANGE:
            start, end = item[1]

            # [0-9]
            if start == ord('0') and end == ord('9'):
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^0-9]',
                        replaced=r'\D',
                        category='char_class_non_digit',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[0-9]',
                        replaced=r'\d',
                        category='char_class_digit',
                    ))

            # [a-z]
            elif start == ord('a') and end == ord('z'):
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^a-z]',
                        replaced=r'\W',
                        category='char_class_non_lower',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[a-z]',
                        replaced=r'\w',
                        category='char_class_lower_expand',
                    ))

            # [A-Z]
            elif start == ord('A') and end == ord('Z'):
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^A-Z]',
                        replaced=r'\W',
                        category='char_class_non_upper',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[A-Z]',
                        replaced=r'\w',
                        category='char_class_upper_expand',
                    ))

        # Отдельные символы (литералы)
        elif item[0] == sre_parse.LITERAL:
            char = chr(item[1])

            # Специальные символы
            special_map = {
                '.': '\.',
                '*': '\*',
                '+': '\+',
                '?': '\?',
                '^': '\^',
                '$': '\$',
                '(': '\(',
                ')': '\)',
                '[': '\[',
                ']': '\]',
                '{': '\{',
                '}': '\}',
                '|': '\|',
                '\\': '\\\\',
            }

            if char in special_map and not is_negated:
                patterns.append(PatternInfo(
                    pattern=f'[{char}]',
                    replaced=special_map[char],
                    category='char_class_special',
                ))

        elif item[0] == sre_parse.CATEGORY:
            cat_value = item[1]

            # \d
            if cat_value == sre_parse.CATEGORY_DIGIT:
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^\\d]',
                        replaced=r'\D',
                        category='inverted_digit',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[\\d]',
                        replaced=r'\d',
                        category='char_class_digit',
                    ))

            # \w
            elif cat_value == sre_parse.CATEGORY_WORD:
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^\\w]',
                        replaced=r'\W',
                        category='inverted_word',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[\\w]',
                        replaced=r'\w',
                        category='char_class_word',
                    ))

            # \s
            elif cat_value == sre_parse.CATEGORY_SPACE:
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^\\s]',
                        replaced=r'\S',
                        category='inverted_space',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[\\s]',
                        replaced=r'\s',
                        category='char_class_space',
                    ))

    elif len(class_ast) == 2:
        item1, item2 = class_ast

        if item1[0] == sre_parse.RANGE and item2[0] == sre_parse.RANGE:
            start1, end1 = item1[1]
            start2, end2 = item2[1]

            # [A-Za-z]
            if (start1 == ord('A') and end1 == ord('Z') and
                    start2 == ord('a') and end2 == ord('z')):
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^A-Za-z]',
                        replaced=r'\W',
                        category='char_class_non_letters',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[A-Za-z]',
                        replaced=r'\w',
                        category='char_class_letters_expand',
                    ))

            # [a-zA-Z]
            elif (start1 == ord('a') and end1 == ord('z') and
                  start2 == ord('A') and end2 == ord('Z')):
                if is_negated:
                    patterns.append(PatternInfo(
                        pattern='[^a-zA-Z]',
                        replaced=r'\W',
                        category='char_class_non_letters',
                    ))
                else:
                    patterns.append(PatternInfo(
                        pattern='[a-zA-Z]',
                        replaced=r'\w',
                        category='char_class_letters_expand',
                    ))

    elif len(class_ast) >= 3:
        # [A-Za-z0-9] -> \w
        has_upper = False
        has_lower = False
        has_digit = False
        has_underscore = False

        for item in class_ast:
            if item[0] == sre_parse.RANGE:
                start, end = item[1]
                if start == ord('A') and end == ord('Z'):
                    has_upper = True
                elif start == ord('a') and end == ord('z'):
                    has_lower = True
                elif start == ord('0') and end == ord('9'):
                    has_digit = True
            elif item[0] == sre_parse.LITERAL and chr(item[1]) == '_':
                has_underscore = True

        # [A-Za-z0-9] -> \w
        if has_upper and has_lower and has_digit and not is_negated:
            patterns.append(PatternInfo(
                pattern='[A-Za-z0-9]',
                replaced=r'\w',
                category='char_class_alnum',
            ))

        # [A-Za-z0-9_] -> \w
        if has_upper and has_lower and has_digit and has_underscore and not is_negated:
            patterns.append(PatternInfo(
                pattern='[A-Za-z0-9_]',
                replaced=r'\w',
                category='char_class_word',
            ))

        # [ \t\n\r\f\v] -> \s
        whitespace_chars = {' ', '\t', '\n', '\r', '\f', '\v'}
        found_whitespace = []

        for item in class_ast:
            if item[0] == sre_parse.LITERAL:
                char = chr(item[1])
                if char in whitespace_chars:
                    found_whitespace.append(char)

        if len(found_whitespace) == len(whitespace_chars):
            if is_negated:
                patterns.append(PatternInfo(
                    pattern='[^ \t\n\r\f\v]',
                    replaced=r'\S',
                    category='char_class_non_whitespace',
                ))
            else:
                patterns.append(PatternInfo(
                    pattern='[ \t\n\r\f\v]',
                    replaced=r'\s',
                    category='char_class_whitespace',
                ))


def extract_single_char_class(class_ast, patterns: List[PatternInfo], is_negated: bool = False):
    if len(class_ast) == 1 and not is_negated:
        item = class_ast[0]
        if item[0] == sre_parse.LITERAL:
            char = chr(item[1])
            patterns.append(PatternInfo(
                pattern=f'[{char}]',
                replaced=re.escape(char) if char in '.[]()*+?^${}||' else char,
                category='single_char_class'
            ))


def extract_empty_non_capturing(value, patterns: List[PatternInfo]):
    group_id, flags, group_pattern = value
    if flags and not group_pattern:
        patterns.append(PatternInfo(
            pattern='(?:)',
            replaced='',
            category='empty_non_capturing'
        ))


def extract_alternatives_to_class(branches, patterns: List[PatternInfo]):
    branch_strings = []
    for branch in branches:
        branch_str = ast_to_literal_string(branch)
        if '...' in branch_str:
            return
        branch_strings.append(branch_str)

    if all(len(s) == 1 for s in branch_strings):
        unique_chars = sorted(set(branch_strings))
        char_class = '[' + ''.join(unique_chars) + ']'
        original = '|'.join(branch_strings)
        patterns.append(PatternInfo(
            pattern=original,
            replaced=char_class,
            category='alternatives_to_class'
        ))


def extract_double_negation(class_ast, patterns: List[PatternInfo], is_negated: bool = False):
    if is_negated and len(class_ast) == 1:
        item = class_ast[0]
        if item[0] == sre_parse.CATEGORY:
            if item[1] == sre_parse.CATEGORY_NON_SPACE:
                patterns.append(PatternInfo(
                    pattern='[^\\S]',
                    replaced=r'\s',
                    category='double_negation'
                ))
            elif item[1] == sre_parse.CATEGORY_NON_DIGIT:
                patterns.append(PatternInfo(
                    pattern='[^\\D]',
                    replaced=r'\d',
                    category='double_negation'
                ))
            elif item[1] == sre_parse.CATEGORY_NON_WORD:
                patterns.append(PatternInfo(
                    pattern='[^\\W]',
                    replaced=r'\w',
                    category='double_negation'
                ))


def extract_redundant_group(ast, patterns: List[PatternInfo]):
    if len(ast) == 1 and ast[0][0] == sre_parse.SUBPATTERN:
        group_id, flags, subpattern = ast[0][1]
        if not flags and group_id is None:
            inner_pattern = ast_to_literal_string(subpattern)
            patterns.append(PatternInfo(
                pattern=f'(?:{inner_pattern})',
                replaced=inner_pattern,
                category='redundant_group'
            ))


def extract_redundant_quantifier(value, patterns: List[PatternInfo]):
    min_count, max_count, subpattern = value
    if min_count == 1 and max_count == 1:
        patterns.append(PatternInfo(
            pattern='{1}',
            replaced='',
            category='redundant_quantifier'
        ))


def extract_empty_groups(value, patterns: List[PatternInfo]):
    group_id, flags, group_pattern = value

    # () -> пустая строка
    if not group_pattern:
        patterns.append(PatternInfo(
            pattern='()',
            replaced='',
            category='empty_group'
        ))

    # (?:) -> пустая строка
    if flags and not group_pattern:
        patterns.append(PatternInfo(
            pattern='(?:)',
            replaced='',
            category='empty_non_capturing'
        ))


def extract_repeated_alternatives(branches, patterns: List[PatternInfo]):
    branch_strings = [ast_to_literal_string(b) for b in branches]

    # a|a -> a
    if len(set(branch_strings)) == 1:
        patterns.append(PatternInfo(
            pattern='|'.join(branch_strings),
            replaced=branch_strings[0],
            category='repeated_alternatives'
        ))

    # a|b|a -> a|b
    if len(set(branch_strings)) < len(branch_strings):
        unique = list(dict.fromkeys(branch_strings))
        patterns.append(PatternInfo(
            pattern='|'.join(branch_strings),
            replaced='|'.join(unique),
            category='duplicate_alternatives'
        ))


def extract_any_char_class(class_ast, patterns: List[PatternInfo], is_negated: bool = False):
    # [\s\S] -> .
    has_space = False
    has_non_space = False

    for item in class_ast:
        if item[0] == sre_parse.CATEGORY:
            if item[1] == sre_parse.CATEGORY_SPACE:
                has_space = True
            elif item[1] == sre_parse.CATEGORY_NON_SPACE:
                has_non_space = True

    if has_space and has_non_space and not is_negated:
        patterns.append(PatternInfo(
            pattern='[\\s\\S]',
            replaced='.',
            category='any_character'
        ))

    # [\d\D] -> .
    has_digit = False
    has_non_digit = False

    for item in class_ast:
        if item[0] == sre_parse.CATEGORY:
            if item[1] == sre_parse.CATEGORY_DIGIT:
                has_digit = True
            elif item[1] == sre_parse.CATEGORY_NON_DIGIT:
                has_non_digit = True

    if has_digit and has_non_digit and not is_negated:
        patterns.append(PatternInfo(
            pattern='[\\d\\D]',
            replaced='.',
            category='any_character_digit'
        ))


def extract_empty_string_anchor(ast, patterns: List[PatternInfo]):
    if len(ast) == 2:
        if ast[0][0] == sre_parse.AT and ast[0][1] == sre_parse.AT_BEGINNING:
            if ast[1][0] == sre_parse.AT and ast[1][1] == sre_parse.AT_END:
                patterns.append(PatternInfo(
                    pattern='^$',
                    replaced='',
                    category='empty_string'
                ))


def extract_category(category_value, patterns: List[PatternInfo]):
    if category_value == sre_parse.CATEGORY_DIGIT:
        patterns.append(PatternInfo(
            pattern=r'\d',
            replaced=r'\d',
            category='shorthand_digit'
        ))

    elif category_value == sre_parse.CATEGORY_WORD:
        patterns.append(PatternInfo(
            pattern=r'\w',
            replaced=r'\w',
            category='shorthand_word'
        ))

    elif category_value == sre_parse.CATEGORY_SPACE:
        patterns.append(PatternInfo(
            pattern=r'\s',
            replaced=r'\s',
            category='shorthand_space'
        ))

    elif category_value == sre_parse.CATEGORY_UNICODE_LETTER:
        patterns.append(PatternInfo(
            pattern=r'\p{L}',
            replaced=r'\w',
            category='unicode_letter'
        ))

    elif category_value == sre_parse.CATEGORY_UNICODE_DIGIT:
        patterns.append(PatternInfo(
            pattern=r'\p{N}',
            replaced=r'\d',
            category='unicode_number'
        ))

    elif category_value == sre_parse.CATEGORY_UNICODE_SPACE:
        patterns.append(PatternInfo(
            pattern=r'\p{Z}',
            replaced=r'\s',
            category='unicode_space'
        ))


def extract_quantifier(value, patterns: List[PatternInfo]):
    min_count, max_count, subpattern = value

    if min_count == 1 and max_count == LIMIT:
        patterns.append(PatternInfo(
            pattern='{1,}',
            replaced='+',
            category='quantifier_one_or_more'
        ))
    elif min_count == 0 and max_count == LIMIT:
        patterns.append(PatternInfo(
            pattern='{0,}',
            replaced='*',
            category='quantifier_zero_or_more'
        ))
    elif min_count == 0 and max_count == 1:
        patterns.append(PatternInfo(
            pattern='{0,1}',
            replaced='?',
            category='quantifier_optional'
        ))


def extract_nested_quantifiers(value, patterns: List[PatternInfo]):
    min_count, max_count, subpattern = value

    if len(subpattern) == 1 and subpattern[0][0] in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
        inner_op = subpattern[0][0]
        inner_min, inner_max, inner_sub = subpattern[0][1]
        outer_pattern = ast_to_literal_string(subpattern[0][1])
        inner_pattern = ast_to_literal_string(inner_sub)
        is_outer_lazy = (subpattern[0][0] == sre_parse.MIN_REPEAT)
        is_inner_lazy = (inner_op == sre_parse.MIN_REPEAT)

        if inner_min == min_count and inner_max == max_count:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern}){outer_pattern}',
                replaced=f'{inner_pattern}+' if min_count == 1 and max_count == LIMIT else f'{inner_pattern}*' if min_count == 0 and max_count == LIMIT else f'{inner_pattern}?' if min_count == 0 and max_count == 1 else f'{inner_pattern}{{{min_count},{max_count}}}',
                category='nested_quantifier_same'
            ))

        elif inner_min == 0 and inner_max == LIMIT and min_count == 1 and max_count == LIMIT:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})*+',
                replaced=f'{inner_pattern}*',
                category='nested_quantifier_star_plus'
            ))

        elif inner_min == 1 and inner_max == LIMIT and min_count == 0 and max_count == LIMIT:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})+*',
                replaced=f'{inner_pattern}*',
                category='nested_quantifier_plus_star'
            ))

        elif inner_min == 0 and inner_max == 1 and min_count == 1 and max_count == LIMIT:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})?+',
                replaced=f'{inner_pattern}?',
                category='nested_quantifier_optional_plus'
            ))

        elif inner_min == 1 and inner_max == LIMIT and min_count == 0 and max_count == 1:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})+?',
                replaced=f'{inner_pattern}*',
                category='nested_quantifier_plus_optional'
            ))

        elif inner_min == 0 and inner_max == 1 and min_count == 0 and max_count == LIMIT:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})?*',
                replaced=f'{inner_pattern}*',
                category='nested_quantifier_optional_star'
            ))

        elif inner_min == 0 and inner_max == LIMIT and min_count == 0 and max_count == 1:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})*?',
                replaced=f'{inner_pattern}*',
                category='nested_quantifier_star_optional'
            ))

        elif inner_min > 0 and inner_max < LIMIT and min_count > 0 and max_count < LIMIT:
            new_min = inner_min * min_count
            new_max = inner_max * max_count
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern}){{{inner_min},{inner_max}}}{{{min_count},{max_count}}}',
                replaced=f'{inner_pattern}{{{new_min},{new_max}}}',
                category='nested_quantifier_multiply'
            ))

        elif inner_min > 0 and inner_max == LIMIT and min_count > 0 and max_count == LIMIT:
            new_min = inner_min * min_count
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern}){{{inner_min},}}{{{min_count},}}',
                replaced=f'{inner_pattern}{{{new_min},}}',
                category='nested_quantifier_multiply_unbounded'
            ))

        elif inner_min > 0 and inner_max < LIMIT and min_count > 0 and max_count == LIMIT:
            new_min = inner_min * min_count
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern}){{{inner_min},{inner_max}}}{{{min_count},}}',
                replaced=f'{inner_pattern}{{{new_min},}}',
                category='nested_quantifier_multiply_left_bounded'
            ))

        elif inner_min > 0 and inner_max == LIMIT and min_count > 0 and max_count < LIMIT:
            new_min = inner_min * min_count
            new_max = inner_max if inner_max == LIMIT else inner_max * max_count
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern}){{{inner_min},}}{{{min_count},{max_count}}}',
                replaced=f'{inner_pattern}{{{new_min},{new_max}}}',
                category='nested_quantifier_multiply_right_bounded'
            ))

        if is_inner_lazy:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})?+?',
                replaced=f'{inner_pattern}??',
                category='nested_quantifier_lazy'
            ))

        if is_outer_lazy:
            patterns.append(PatternInfo(
                pattern=f'({inner_pattern})+??',
                replaced=f'{inner_pattern}+?',
                category='nested_quantifier_outer_lazy'
            ))


def extract_alternatives(branch_ast, original_regex: str, patterns: List[PatternInfo]):
    _, branches = branch_ast

    if len(branches) == 2:
        branch1_str = ast_to_literal_string(branches[0])
        branch2_str = ast_to_literal_string(branches[1])

        # http|https -> https?
        if branch1_str == 'http' and branch2_str == 'https':
            patterns.append(PatternInfo(
                pattern='http|https',
                replaced='https?',
                category='alternatives_prefix'
            ))

        # (a)? -> a?
        elif branch1_str == '' and branch2_str in ALPHABET:
            patterns.append(PatternInfo(
                pattern=f'({branch2_str})?',
                replaced=f'{branch2_str}?',
                category='alternatives_optional'
            ))

        # aa|a -> aa?
        elif branch1_str == ''.join([branch2_str,branch2_str]) and branch2_str in ALPHABET:
            patterns.append(PatternInfo(
                pattern=f'{branch2_str}{branch2_str}|{branch2_str}',
                replaced=f'{branch2_str}{branch2_str}?',
                category='alternatives_optional'
            ))

        # a|aa -> aa?
        elif branch1_str in ALPHABET and branch2_str == ''.join([branch1_str,branch1_str]):
            patterns.append(PatternInfo(
                pattern=f'{branch1_str}{branch1_str}|{branch1_str}',
                replaced=f'{branch1_str}{branch1_str}?',
                category='alternatives_optional'
            ))

    for branch in branches:
        extract_patterns_from_ast(branch, original_regex, patterns)


def extract_patterns_from_ast(
        ast,
        original_regex: str,
        patterns: List[PatternInfo]
):
    extract_empty_string_anchor(ast, patterns)

    for op, value in ast:
        if op == sre_parse.IN:
            is_negated = False
            if value and value[0][0] == sre_parse.NEGATE:
                is_negated = True
            extract_char_class(value, patterns)
            extract_single_char_class(value, patterns, is_negated)
            extract_double_negation(value, patterns, is_negated)
            extract_any_char_class(value, patterns, is_negated)

        elif op == sre_parse.CATEGORY:
            extract_category(value, patterns)

        elif op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            extract_nested_quantifiers(value, patterns)
            extract_quantifier(value, patterns)
            extract_redundant_quantifier(value, patterns)
            _, _, subpattern = value
            extract_patterns_from_ast(subpattern, original_regex, patterns)

        elif op == sre_parse.BRANCH:
            _, branches = value
            extract_alternatives(value, original_regex, patterns)
            extract_repeated_alternatives(branches, patterns)
            extract_alternatives_to_class(branches, patterns)
            for branch in branches:
                extract_patterns_from_ast(branch, original_regex, patterns)

        elif op == sre_parse.SUBPATTERN:
            extract_redundant_group([(op, value)], patterns)
            extract_empty_groups(value, patterns)
            extract_empty_non_capturing(value, patterns)
            _, _, group_pattern = value
            extract_patterns_from_ast(group_pattern, original_regex, patterns)

        elif op == sre_parse.NOT_LITERAL:
            pass

        else:
            pass


def extract_all_patterns(regex_str: str) -> List[PatternInfo]:
    patterns = []

    try:
        parsed = sre_parse.parse(regex_str)
        extract_patterns_from_ast(parsed, regex_str, patterns)
    except:
        pass

    return patterns


def collect_statistics(dataset: List[str], min_frequency: int = 10) -> Tuple[Dict, Dict]:
    stats = defaultdict(int)
    pattern_map = {}

    for regex in dataset:
        patterns: List[PatternInfo] = extract_all_patterns(regex)

        for pattern_info in patterns:
            key = (pattern_info.pattern, pattern_info.replaced, pattern_info.category)
            stats[key] += 1

            if key not in pattern_map:
                pattern_map[key] = pattern_info

    frequent = {key: count for key, count in stats.items() if count >= min_frequency}

    return frequent, pattern_map


def generate_replacements(stats: Dict, min_frequency: int = 10) -> List[Tuple[str, str, str, int]]:
    replacements = []

    for (original, replaced, category), count in stats.items():
        if count >= min_frequency:
            replacements.append((original, replaced, category, count))

    replacements.sort(key=lambda x: -x[3])

    return replacements


def format_replacements(
        replacements: List[Tuple[str, str, str, int]],
        is_need_stats: bool = False
) -> str:
    lines = []
    for original, replaced, category, count in replacements:
        if is_need_stats:
            lines.append(f"{original} -> {replaced}  # {category}, frequency: {count}")
        else:
            lines.append(f"{original} -> {replaced}")

    return "\n".join(lines)


def generate(
        dataset: List[str],
        min_frequency: int,
        logger,
        path: Path = Path('generated_replacements.txt'),
        verbose: bool = False,
):
    if verbose:
        logger.debug(f"Dataset size: {len(dataset)}")
        logger.debug("Stats calculating...")
    stats, _ = collect_statistics(dataset, min_frequency)

    logger.info(f"Unique replacements founded: {len(stats)}")

    if verbose:
        logger.debug(f"Unique constructions: {len(stats)}")
        logger.debug(f"(>= {min_frequency}): {len([k for k in stats if stats[k] >= min_frequency])}")
        logger.debug("Replacements generating...")
    replacements = generate_replacements(stats, min_frequency)

    with open(path, 'w', encoding='utf-8') as f:
        f.write(format_replacements(replacements, verbose))

    if verbose:
        logger.debug("Top-5 replacements:")
        for i, (original, simplified, category, count) in enumerate(replacements[:5]):
            logger.debug(f"{i + 1}. {original} -> {simplified} (frequency: {count})")

    return replacements
