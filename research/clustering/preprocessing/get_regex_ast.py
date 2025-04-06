import re
try:
    import re._parser as sre_parse
except ImportError:
    # Python < 3.11
    from re import sre_parse


from logger import logger


class SreParser:
    def __init__(self, verbose: bool = False):
        self.errors = 0
        self.verbose = verbose

        self.reg_clean = re.compile(r'\(LITERAL\,\s\d*\)')

    def __repr__(self):
        return 'sre_parser'

    def parse(self, regex):
        try:
            return sre_parse.parse(regex)
        except Exception as e:
            if self.verbose:
                logger.warning(f'This expression <{regex}> does not written by python flavor: {e}')
            self.errors += 1
            return None

    def postprocess(self, ast):
        try:
            clean_ast = self.reg_clean.sub(
                string=str(ast),
                repl='LITERAL'
            )
            return clean_ast
        except Exception as e:
            if self.verbose:
                logger.warning(f'Error while this ast <{ast}> cleaning: {e}')
            return ast

    def parse_list(
        self,
        regex_list,
        dialects,
        is_need_postprocessing: bool = True,
    ):
        new_dialects = []
        ast = []
        for i, regex in enumerate(regex_list):
            parsed_regex = self.parse(regex)
            if parsed_regex:
                if is_need_postprocessing:
                    ast.append(
                        self.postprocess(parsed_regex)
                    )
                else:
                    ast.append(
                        str(parsed_regex)
                    )
                new_dialects.append(dialects[i])
        if self.errors > 0:
            logger.warning(f'{self.errors} regexes does not written by python flavor')
        self.errors = 0
        return ast, new_dialects


if __name__ == '__main__':
    parser = SreParser(verbose=True)
    regex = input('Input regex: ')
    ast = parser.parse(regex)
    print(f'AST: {ast}')
    processed_ast = parser.postprocess(ast)
    print(f'Post AST: {processed_ast}')
