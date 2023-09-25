import dataclasses as dc
import typing as t

import tenacity

import superduperdb as s

ExceptionTypes = t.Union[t.Type[BaseException], t.Tuple[t.Type[BaseException], ...]]


@dc.dataclass
class Retry:
    """
    Retry a function until it succeeds.

    This is a thin wrapper around the tenacity retry library, using our configs.
    :param exception_types: The exception types to retry on.
    :param cfg: The retry config.
    """

    exception_types: ExceptionTypes
    cfg: t.Optional[s.config.Retry] = None

    def __call__(self, f: t.Callable) -> t.Any:
        cfg = self.cfg or s.CFG.apis.retry
        retry = tenacity.retry_if_exception_type(self.exception_types)
        stop = tenacity.stop_after_attempt(cfg.stop_after_attempt)
        wait = tenacity.wait_exponential(
            max=cfg.wait_max,
            min=cfg.wait_min,
            multiplier=cfg.wait_multiplier,
        )
        retrier = tenacity.retry(retry=retry, stop=stop, wait=wait)
        return retrier(f)
