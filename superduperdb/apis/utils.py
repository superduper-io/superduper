from tenacity import retry_if_exception_type, retry, stop_after_attempt, wait_exponential

from superduperdb.apis import api_cf


class DoRetry:
    def __init__(self, exception_types=()):
        self.exception_types = exception_types

    def __call__(self, f):
        return retry(
            retry=retry_if_exception_type(self.exception_types),
            stop=stop_after_attempt(api_cf.get('n_retries', 2)),
            wait = wait_exponential(multiplier=1, min=4, max=10)
        )(f)
