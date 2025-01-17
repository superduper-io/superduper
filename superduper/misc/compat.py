"""Functions from later standard libraries not available in Python 3.8."""

# TODO not needed

from functools import lru_cache

__all__ = ('cache',)


# Implements functools.cache from Python 3.9
def cache(user_function, /):
    """Simple cache decorator.

    This is a simple cache decorator that can be used to cache the results of
    a function call. It does not have any of the advanced features of
    functools.lru_cache.

    :param user_function: Function to cache
    """
    return lru_cache(maxsize=None)(user_function)
