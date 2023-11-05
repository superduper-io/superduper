import os
from sys import stderr, stdout

from loguru import logger
from loki_logger_handler.loki_logger_handler import (LoguruFormatter,
                                                     LokiLoggerHandler)

from tqdm import tqdm

from superduperdb.base.config import LogLevel, LogType

from .configs import CFG

__all__ = ('Logging',)


class Logging:
    if CFG.logging.type == LogType.LOKI:  # Send logs to Loki
        custom_handler = LokiLoggerHandler(
            url=os.environ["LOKI_URI"],
            labels={"application": "Test", "environment": "Develop"},
            labelKeys={},
            timeout=10,
            defaultFormatter=LoguruFormatter(),
        )

        logger.configure(handlers=[{"sink": custom_handler, "serialize": True}])
    else:
        # Replace default logger with a custom SuperDuper format.
        logger.remove()

        fmt = (
            " <green> {time:YYYY-MMM-DD HH:mm:ss}</green> | <level>{level: <8}</level> |"
            " <cyan>{thread.name}</cyan>:<cyan>{name}</cyan>:<cyan>{line}</cyan> |"
            " <level> {message} | {extra} </level>"
        )

        # DEBUG until WARNING are sent to stdout.
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format=fmt,
            level=CFG.logging.level,
            filter=lambda record: record["level"].no < 40,
            colorize=True,
        )

        # ERROR and above sent to stderr
        # https://loguru.readthedocs.io/en/stable/api/logger.html
        logger.add(
            stderr,
            format=fmt,
            level=LogLevel.ERROR,
            colorize=True,
        )

    # Set Multi-Key loggers
    # Example: logging.info("param 1", "param 2", ..)
    def multikey_debug(*args):
        logger.opt(depth=1).debug(" ".join(map(str, args)))

    def multikey_info(*args):
        logger.opt(depth=1).info(" ".join(map(str, args)))

    def multikey_success(*args):
        logger.opt(depth=1).success(" ".join(map(str, args)))

    def multikey_warn(*args):
        logger.opt(depth=1).warning(" ".join(map(str, args)))

    def multikey_error(*args):
        logger.opt(depth=1).error(" ".join(map(str, args)))

    debug = multikey_debug
    info = multikey_info
    success = multikey_success
    warn = multikey_warn
    error = multikey_error
    bind = logger.bind
