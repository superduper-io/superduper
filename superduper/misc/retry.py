import dataclasses as dc
import functools
import typing as t

import tenacity

import superduper as s

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


def db_retry(connector='databackend'):
    """Helper method to retry methods with database calls.

    :param connector: Connector of the datalayer instance.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                error_message = str(e).lower()
                if 'expire' in error_message or 'token' in error_message:
                    s.logging.warn(
                        f"Token expiration detected: {e}. Attempting to reconnect..."
                    )
                    self.databackend.reconnect()
                    return func(self, *args, **kwargs)
                else:
                    raise e

        return wrapper

    return decorator
