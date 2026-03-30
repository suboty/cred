import sys
import logging


def get_logger():
    logger_ = logging.getLogger('root')

    logger_.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger_.addHandler(handler)

    return logger_


logger = get_logger()
