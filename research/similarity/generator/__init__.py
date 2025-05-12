import re
from pathlib import Path
from typing import Union, List


class Generator:
    def __init__(
            self,
            path_to_replacements: Path = Path('generator')
    ):
        self.equivalent_replacements, self.reg_equivalent_replacements = self.read_replacements(
            Path(path_to_replacements, 'replacements')
        )

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
    def run_repl(strings, repls, reg_repls):

        repl_res = {}
        for i in range(len(strings)):
            repl_res[strings[i]] = []
            for replacement in repls:
                if replacement[0] in strings[i]:
                    repl_res[strings[i]].append(strings[i].replace(
                        replacement[0],
                        replacement[1]
                    ))
            for replacement in reg_repls:
                _res = re.sub(
                    pattern=replacement[0],
                    string=strings[i],
                    repl=replacement[1]
                )
                if _res != strings[i]:
                    repl_res[strings[i]].append(_res)
        return repl_res

    def __call__(
        self,
        regex_list: List,
        need_remove_comments: bool = False,
    ):

        if need_remove_comments:
            regex_list = [self.remove_comment_group(x) for x in regex_list]

        regex_list = self.run_repl(
            strings=regex_list,
            repls=self.equivalent_replacements,
            reg_repls=self.reg_equivalent_replacements,
        )

        return regex_list


if __name__ == '__main__':
    g = Generator(Path('.'))

    res = g(
        regex_list=['[0-9]*', '[0-9]abc(s)?']
    )

    print(f'Generator output: {res}')
