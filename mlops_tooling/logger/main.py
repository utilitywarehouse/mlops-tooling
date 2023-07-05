import logging
import colorlog
import sys


def get_logger(name: str) -> logging.Logger:
    """
    Creates a logger to use for logging.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = create_handler()
        logger.addHandler(handler)

    return logger


def create_handler():
    """
    Creates a handler identical to Optuna's for consistent logging.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(create_default_formatter())
    return handler


def create_default_formatter():
    """
    Colours a logger output to make it more readable, matching that of Optuna's for consistency.
    """
    return colorlog.ColoredFormatter(
        "%(log_color)s[%(levelname)1.1s %(asctime)s]%(reset)s %(message)s"
    )
