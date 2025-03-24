try:
    import re._parser as sre_parse
except ImportError:
    # Python < 3.11
    from re import sre_parse


from logger import logger


class SreParser:
    def __repr__(self):
        return 'sre_parser'

    @staticmethod
    def parse(regex):
        try:
            return sre_parse.parse(regex)
        except Exception as e:
            logger.warning(f'This expression <{regex}> does not written by python flavor: {e}')
            return None

    @staticmethod
    def parse_list(regex_list, dialects):
        new_dialects = []
        ast = []
        for i, regex in enumerate(regex_list):
            parsed_regex = SreParser.parse(regex)
            if parsed_regex:
                ast.append(str(regex))
                new_dialects.append(dialects[i])
        return ast, new_dialects
