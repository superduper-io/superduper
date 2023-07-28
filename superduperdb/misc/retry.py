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
    """

    exception_types: ExceptionTypes
    cfg: s.config.Retry = dc.field(default_factory=lambda: s.CFG.apis.retry)

    def __call__(self, f: t.Callable) -> t.Any:
        retry = tenacity.retry_if_exception_type(self.exception_types)
        stop = tenacity.stop_after_attempt(self.cfg.stop_after_attempt)
        wait = tenacity.wait_exponential(
            max=self.cfg.wait_max,
            min=self.cfg.wait_min,
            multiplier=self.cfg.wait_multiplier,
        )
        retrier = tenacity.retry(retry=retry, stop=stop, wait=wait)
        return retrier(f)
