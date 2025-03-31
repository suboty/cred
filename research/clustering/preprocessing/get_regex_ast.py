try:
    import re._parser as sre_parse
except ImportError:
    # Python < 3.11
    from re import sre_parse


from logger import logger


class SreParser:
    def __init__(self):
        self.errors = 0
        self.verbose = False

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

    def parse_list(self, regex_list, dialects):
        new_dialects = []
        ast = []
        for i, regex in enumerate(regex_list):
            parsed_regex = self.parse(regex)
            if parsed_regex:
                ast.append(str(regex))
                new_dialects.append(dialects[i])
        if self.errors > 0:
            logger.warning(f'{self.errors} regexes does not written by python flavor')
        self.errors = 0
        return ast, new_dialects


if __name__ == '__main__':
    parser = SreParser()
    regex = input('Input regex: ')
    print(str(parser.parse(regex)))
