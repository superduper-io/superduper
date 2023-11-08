import os
import socket
from sys import stderr

from loguru import logger
from loki_logger_handler.loki_logger_handler import LoguruFormatter, LokiLoggerHandler
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

        # Enrich logger with additional information.
        logger.configure(
            extra={
                "hostname": socket.gethostname(),
            }
        )

        fmt = (
            " <green> {time:YYYY-MMM-DD HH:mm:ss}</green>"
            " | <level>{level: <8} </level> "
            " | <cyan>{extra[hostname]}</cyan>"
            " | <cyan>{name}</cyan>:<cyan>{line: <4}</cyan> "
            " | <level> {message} </level>"
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

    # Set log levels
    debug = logger.debug
    info = logger.info
    success = logger.success
    warn = logger.warning
    error = logger.error
