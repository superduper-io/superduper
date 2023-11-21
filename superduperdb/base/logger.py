import os
import socket
import uuid
from sys import stderr

from loguru import logger
from loki_logger_handler.loki_logger_handler import LoguruFormatter, LokiLoggerHandler
from tqdm import tqdm

from superduperdb.base.config import LogLevel, LogType

from .configs import CFG

__all__ = ('Logging',)


class Logging:
    if CFG.logging_type == LogType.LOKI:  # Send logs to Loki
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

        # Enrich logger with additional information.
        logger.configure(
            extra={
                "hostname": socket.gethostname(),
                "session_id": str(uuid.uuid4()),
            }
        )

        fmt = (
            "<green> {time:YYYY-MMM-DD HH:mm:ss.SS}</green>"
            "| <level>{level: <8}</level> "
            "| <cyan>{extra[hostname]: <8}</cyan>"
            "| <cyan>{extra[session_id]}</cyan>"
            "| <cyan>{name}</cyan>:<cyan>{line: <4}</cyan> "
            "| <level>{message}</level>"
        )

        # DEBUG until WARNING are sent to stdout.
        logger.add(
            lambda msg: tqdm.write(msg, end=""),
            format=fmt,
            level=CFG.log_level,
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
    @staticmethod
    def multikey_debug(msg: str, *args):
        logger.opt(depth=1).debug(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_info(msg: str, *args):
        logger.opt(depth=1).info(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_success(msg: str, *args):
        logger.opt(depth=1).success(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_warn(msg: str, *args):
        logger.opt(depth=1).warning(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_error(msg: str, *args):
        logger.opt(depth=1).error(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_exception(msg: str, *args, e=None):
        logger.opt(depth=1, exception=e).error(" ".join(map(str, (msg, *args))))

    debug = multikey_debug
    info = multikey_info
    success = multikey_success
    warn = multikey_warn
    error = multikey_error
    exception = multikey_exception
