import os
import socket
from sys import stderr

from loguru import logger
from tqdm import tqdm

from superduper.base.config import LogLevel
from superduper.base.configs import CFG

__all__ = ('Logging',)

PROJECT_ROOT = os.getcwd()


def patcher(record):
    """Patch the logger to add the relative path of the file.

    :param record: The log record.
    """
    abs_path = record["file"].path
    if abs_path.startswith(PROJECT_ROOT):
        rel_path = os.path.relpath(abs_path, PROJECT_ROOT)
    else:
        rel_path = os.path.basename(abs_path)
    record["extra"]["relpath"] = rel_path


logger = logger.patch(patcher)
logger.level("USER", no=35)


class Logging:
    """Logging class to handle logging for the superduper.io # noqa."""

    # Replace default logger with a custom SuperDuper format.
    logger.remove()

    # Enrich logger with additional information.
    logger.configure(
        extra={
            "hostname": socket.gethostname(),
        }
    )

    colorize = CFG.log_colorize
    if colorize:
        fmt = (
            "<green>{time:YYYY-MMM-DD HH:mm:ss.SS}</green>"
            "| <level>{level: <8}</level> "
            "{hostname}"
            "| <cyan>{extra[relpath]}</cyan>:<cyan>{line}</cyan>"
            "| <level>{message}</level>"
        )
        if CFG.log_hostname:
            fmt = fmt.replace(
                "{hostname}",
                "| <cyan>{extra[hostname]: <8}</cyan>",
            )
        else:
            fmt = fmt.replace("{hostname}", "")
    else:
        fmt = (
            "{time:YYYY-MMM-DD HH:mm:ss.SS}"
            "| {level: <8} "
            "| {extra[relpath]}:{line} "
            "| {message}"
        )
    # DEBUG until WARNING are sent to stdout.
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=fmt,
        level=CFG.log_level,
        filter=lambda record: record["level"].no < 40
        and record["level"].name != 'USER',
        colorize=colorize,
    )

    USER_FMT = "{message}"  # just the text (colour OK)
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        format=USER_FMT,
        level="USER",  # min-level is USER…
        filter=lambda r: r["level"].name == "USER",  # …and *only* USER passes
        colorize=True,  # or use `colorize`
    )

    # ERROR and above sent to stderr
    # https://loguru.readthedocs.io/en/stable/api/logger.html
    logger.add(
        stderr,
        format=fmt,
        level=LogLevel.ERROR,
        colorize=colorize,
    )

    # Set Multi-Key loggers
    # Example: logging.info("param 1", "param 2", ..)
    @staticmethod
    def multikey_debug(msg: str, *args):
        """Log a message with the DEBUG level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        """
        logger.opt(depth=1).debug(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_info(msg: str, *args, depth: int = 1):
        """Log a message with the INFO level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        :param depth: The depth of the log message in the stack.
        """
        logger.opt(depth=depth).info(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_success(msg: str, *args):
        """Log a message with the SUCCESS level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        """
        logger.opt(depth=1).success(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_warn(msg: str, *args):
        """Log a message with the WARNING level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        """
        logger.opt(depth=1).warning(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_error(msg: str, *args):
        """Log a message with the ERROR level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        """
        logger.opt(depth=1).error(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_exception(msg: str, *args, e=None):
        """Log a message with the ERROR level.

        e.g. logger.exception("An error occurred", e)

        :param msg: The message to log.
        :param args: Additional arguments to log.
        :param e: The exception to log.
        """
        logger.opt(depth=1, exception=e).error(" ".join(map(str, (msg, *args))))

    @staticmethod
    def multikey_user(msg: str, *args):
        """Log a message with the USER level.

        :param msg: The message to log.
        :param args: Additional arguments to log.
        """
        logger.opt(depth=1).log("USER", " ".join(map(str, (msg, *args))))

    user = multikey_user
    debug = multikey_debug
    info = multikey_info
    success = multikey_success
    warn = multikey_warn
    error = multikey_error
    exception = multikey_exception
