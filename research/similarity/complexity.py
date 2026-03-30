import sre_parse
from typing import List, Tuple


def calculate_ast_complexity(ast) -> int:
    """Calculates the complexity of the AST (number of nodes)"""
    complexity = 1

    for op, value in ast:
        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            _, _, subpattern = value
            complexity += calculate_ast_complexity(subpattern)
        elif op == sre_parse.BRANCH:
            _, branches = value
            for branch in branches:
                complexity += calculate_ast_complexity(branch)
        elif op == sre_parse.SUBPATTERN:
            _, _, subpattern = value
            complexity += calculate_ast_complexity(subpattern)
        elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
            _, subpattern = value
            complexity += calculate_ast_complexity(subpattern)
        elif op == sre_parse.IN:
            for item in value:
                if item[0] == sre_parse.CATEGORY:
                    complexity += 1
                elif item[0] == sre_parse.RANGE:
                    complexity += 1
                elif item[0] == sre_parse.LITERAL:
                    complexity += 1
                else:
                    complexity += 1
        elif op == sre_parse.CATEGORY:
            complexity += 1
        elif op == sre_parse.LITERAL:
            complexity += 1
        elif op == sre_parse.AT:
            complexity += 1
        elif op == sre_parse.GROUPREF:
            complexity += 1

    return complexity


def calculate_depth(ast, current_depth=0) -> int:
    """Calculates the maximum nesting depth of an AST"""
    max_depth = current_depth

    for op, value in ast:
        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            _, _, subpattern = value
            depth = calculate_depth(subpattern, current_depth + 1)
            max_depth = max(max_depth, depth)
        elif op == sre_parse.BRANCH:
            _, branches = value
            for branch in branches:
                depth = calculate_depth(branch, current_depth + 1)
                max_depth = max(max_depth, depth)
        elif op == sre_parse.SUBPATTERN:
            _, _, subpattern = value
            depth = calculate_depth(subpattern, current_depth + 1)
            max_depth = max(max_depth, depth)
        elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
            _, subpattern = value
            depth = calculate_depth(subpattern, current_depth + 1)
            max_depth = max(max_depth, depth)
        elif op == sre_parse.IN:
            max_depth = max(max_depth, current_depth + 1)

    return max_depth


def count_nodes_by_type(ast, node_counts: dict = None) -> dict:
    """Counts the number of nodes of each type"""
    if node_counts is None:
        node_counts = {}

    op_names = {
        sre_parse.LITERAL: 'literal',
        sre_parse.NOT_LITERAL: 'not_literal',
        sre_parse.CATEGORY: 'category',
        sre_parse.IN: 'char_class',
        sre_parse.MAX_REPEAT: 'max_repeat',
        sre_parse.MIN_REPEAT: 'min_repeat',
        sre_parse.BRANCH: 'branch',
        sre_parse.SUBPATTERN: 'subpattern',
        sre_parse.ASSERT: 'assert',
        sre_parse.ASSERT_NOT: 'assert_not',
        sre_parse.AT: 'anchor',
        sre_parse.GROUPREF: 'groupref',
        sre_parse.RANGE: 'range',
    }

    for op, value in ast:
        op_name = op_names.get(op, f'unknown_{op}')
        node_counts[op_name] = node_counts.get(op_name, 0) + 1

        if op in (sre_parse.MAX_REPEAT, sre_parse.MIN_REPEAT):
            _, _, subpattern = value
            count_nodes_by_type(subpattern, node_counts)
        elif op == sre_parse.BRANCH:
            _, branches = value
            for branch in branches:
                count_nodes_by_type(branch, node_counts)
        elif op == sre_parse.SUBPATTERN:
            _, _, subpattern = value
            count_nodes_by_type(subpattern, node_counts)
        elif op in (sre_parse.ASSERT, sre_parse.ASSERT_NOT):
            _, subpattern = value
            count_nodes_by_type(subpattern, node_counts)
        elif op == sre_parse.IN:
            for item in value:
                if item[0] == sre_parse.RANGE:
                    node_counts['range'] = node_counts.get('range', 0) + 1
                elif item[0] == sre_parse.CATEGORY:
                    node_counts['category_in_class'] = node_counts.get('category_in_class', 0) + 1
                elif item[0] == sre_parse.LITERAL:
                    node_counts['literal_in_class'] = node_counts.get('literal_in_class', 0) + 1
                elif item[0] == sre_parse.NEGATE:
                    node_counts['negate'] = node_counts.get('negate', 0) + 1

    return node_counts


def measure_complexity_reduction(original_regex: str, replacements: List[Tuple[str, str, str, int]]) -> dict:
    """
    Measures how much a regular expression has been simplified after substitutions have been applied.
    """
    try:
        original_ast = sre_parse.parse(original_regex)
        original_complexity = calculate_ast_complexity(original_ast)
        original_depth = calculate_depth(original_ast)
        original_nodes = count_nodes_by_type(original_ast)
    except:
        return {
            'error': f'Failed to parse original regex: {original_regex}',
            'original_regex': original_regex,
            'simplified_regex': original_regex
        }

    simplified = original_regex
    applied_replacements = []

    for original, replaced, category, _ in replacements:
        if original in simplified:
            new_simplified = simplified.replace(original, replaced)
            if new_simplified != simplified:
                applied_replacements.append({
                    'original': original,
                    'replaced': replaced,
                    'category': category
                })
                simplified = new_simplified

    try:
        simplified_ast = sre_parse.parse(simplified)
        simplified_complexity = calculate_ast_complexity(simplified_ast)
        simplified_depth = calculate_depth(simplified_ast)
        simplified_nodes = count_nodes_by_type(simplified_ast)
    except:
        return {
            'error': f'Failed to parse simplified regex: {simplified}',
            'original_regex': original_regex,
            'simplified_regex': simplified,
            'original_complexity': original_complexity,
            'original_depth': original_depth
        }

    complexity_reduction = ((original_complexity - simplified_complexity)
                            / original_complexity) * 100 if original_complexity > 0 else 0
    depth_reduction = ((original_depth - simplified_depth)
                       / original_depth) * 100 if original_depth > 0 else 0

    return {
        'original_complexity': original_complexity,
        'simplified_complexity': simplified_complexity,
        'complexity_reduction': complexity_reduction,
        'original_depth': original_depth,
        'simplified_depth': simplified_depth,
        'depth_reduction': depth_reduction,
        'original_nodes': original_nodes,
        'simplified_nodes': simplified_nodes,
        'original_regex': original_regex,
        'simplified_regex': simplified,
        'replacements_applied': applied_replacements
    }
