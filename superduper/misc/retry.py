import dataclasses as dc
import functools
import time
import typing as t

import tenacity

import superduper as s
from superduper import logging
from superduper.misc.files import load_secrets

ExceptionTypes = t.Union[t.Type[BaseException], t.Tuple[t.Type[BaseException], ...]]


@dc.dataclass
class Retry:
    """
    Retry a function until it succeeds.

    This is a thin wrapper around the tenacity retry library, using our configs.
    :param exception_types: The exception types to retry on.
    :param cfg: The retry config. If None, uses the default config.
    """

    exception_types: ExceptionTypes
    cfg: t.Optional[s.config.Retry] = None

    def __call__(self, f: t.Callable) -> t.Any:
        """Decorate a function to retry on exceptions.

        Uses the exception types and config provided to the constructor.
        :param f: The function to decorate.
        """
        cfg = self.cfg or s.CFG.retries
        retry = tenacity.retry_if_exception_type(self.exception_types)
        stop = tenacity.stop_after_attempt(cfg.stop_after_attempt)
        wait = tenacity.wait_exponential(
            max=cfg.wait_max,
            min=cfg.wait_min,
            multiplier=cfg.wait_multiplier,
        )
        retrier = tenacity.retry(retry=retry, stop=stop, wait=wait)
        return retrier(f)


def safe_retry(exception_to_check, retries=1, delay=0.3, verbose=1):
    """
    A decorator that retries a function if a specified exception is raised.

    :param exception_to_check: The exception or tuple of exceptions to check.
    :param retries: The maximum number of retries.
    :param delay: Delay between retries in seconds.
    :param verbose: Verbose for logs.
    :return: The result of the decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            while attempt <= retries:
                try:
                    if attempt >= 1:
                        load_secrets()
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    attempt += 1
                    if attempt > retries:
                        if verbose:
                            logging.error(
                                f"Function {func.__name__} failed ",
                                "after {retries} retries.",
                            )
                        raise
                    if verbose:
                        logging.warn(
                            f"Retrying {func.__name__} due to {e}"
                            f", attempt {attempt} of {retries}..."
                        )
                    time.sleep(delay)

        return wrapper

    return decorator
