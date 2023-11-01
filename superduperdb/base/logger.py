from superduperdb.base.config import LogLevel, LogType

from .configs import CFG

__all__ = ('Logging',)


class Logging:
    if CFG.logging.type == LogType.SYSTEM:
        from sys import stderr, stdout

        from loguru import logger

        # Replace default with SuperDuperDB settings
        logger.remove()
        fmt = "<green>{time:YYYY-MMM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> | <cyan>{extra}</cyan> <level> | {message} </level> "

        # Send "ERROR" and below to stdout, with "ERROR" and above copied to stderr
        logger.add(stdout, format=fmt, level=CFG.logging.level)
        logger.add(stderr, format=fmt, level=LogLevel.ERROR)

        # Send "ERROR" and below to stdout, with "ERROR" and above moved to stderr
        # logger.add(sys.stdout, filter=lambda record: record["level"].no < 40, level="INFO")

        # Set log levels
        debug = logger.debug
        info = logger.info
        success = logger.success
        warn = logger.warning
        error = logger.error
    else:
        import os

        from loguru import logger
        from loki_logger_handler.loki_logger_handler import (
            LoguruFormatter,
            LokiLoggerHandler,
        )

        custom_handler = LokiLoggerHandler(
            url=os.environ["LOKI_URI"],
            labels={"application": "Test", "environment": "Develop"},
            labelKeys={},
            timeout=10,
            defaultFormatter=LoguruFormatter(),
        )

        logger.configure(handlers=[{"sink": custom_handler, "serialize": True}])

        # Set log levels
        debug = logger.debug
        info = logger.info
        success = logger.success
        warn = logger.warning
        error = logger.error
