import json
import re
import copy
from pathlib import Path
from collections import Counter
from typing import Union, List


class Replacements:
    def __init__(
            self,
            path_to_replacements: Path = Path('preprocessing')
    ):
        self.equivalent_replacements, self.reg_equivalent_replacements = self.read_replacements(
            Path(path_to_replacements, 'equivalent_replacements')
        )

        self.nearly_equivalent_replacements, self.reg_nearly_equivalent_replacements = self.read_replacements(
            Path(path_to_replacements, 'nearly_equivalent_replacements')
        )

        self.fstats = {
            'eq': [],
            'neq': []
        }

    @staticmethod
    def read_replacements(
            path_to_replacements: Union[Path, str]
    ):
        with open(path_to_replacements, 'r') as repl_file:
            data = repl_file.read().split('\n')
        replacements, reg_replacements = [], []
        for repl in data:
            if ' r->r ' in repl:
                reg_replacements.append(tuple(repl.split(' r->r ')))
            elif ' -> ' in repl:
                replacements.append(tuple(repl.split(' -> ')))
        return replacements, reg_replacements

    @staticmethod
    def remove_comment_group(regex):
        return re.sub(
            pattern=r'\(\?\#[^\)]*\)',
            string=regex,
            repl='',
        )

    @staticmethod
    def make_stats_report(stats):
        mappings = {
            'eq': 'Equivalent Replacements',
            'neq': 'Nearly Equivalent Replacements',
            'rep': 'Usual Replacements',
            'reg_rep': 'Regular Expression Replacements'
        }
        report = {}
        for key in stats.keys():
            report[mappings.get(key)] = {}
            for method in stats[key].keys():
                _counter = Counter(stats[key][method])
                report[mappings.get(key)][mappings.get(method)] = dict(_counter)
        with open(Path('tmp', 'replacements_stats.json'), 'w') as f:
            json.dump(report, f, ensure_ascii=False)

    @staticmethod
    def run_repl(strings, repls, reg_repls):
        stats_list = {
            'rep': [],
            'reg_rep': []
        }

        for i in range(len(strings)):
            for replacement in repls:
                if replacement[0] in strings[i]:
                    stats_list['rep'].append(' -> '.join(replacement))
                    strings[i] = strings[i].replace(
                        replacement[0],
                        replacement[1]
                    )
            for replacement in reg_repls:
                if re.search(pattern=replacement[0], string=strings[i]):
                    stats_list['reg_rep'].append(' r->r '.join(replacement))
                strings[i] = re.sub(
                    pattern=replacement[0],
                    string=strings[i],
                    repl=replacement[1]
                )
        return strings, stats_list

    def __call__(
        self,
        regex_list: List,
        need_equivalent: bool = False,
        need_nearly_equivalent: bool = False,
        need_remove_comments: bool = False,
    ):
        _stats = copy.deepcopy(self.fstats)

        if need_remove_comments:
            regex_list = [self.remove_comment_group(x) for x in regex_list]

        if need_equivalent:
            regex_list, _stats['eq'] = self.run_repl(
                strings=regex_list,
                repls=self.equivalent_replacements,
                reg_repls=self.reg_equivalent_replacements,
            )

        if need_nearly_equivalent:
            regex_list, _stats['neq'] = self.run_repl(
                strings=regex_list,
                repls=self.nearly_equivalent_replacements,
                reg_repls=self.reg_nearly_equivalent_replacements,
            )

        self.make_stats_report(_stats)

        return regex_list
