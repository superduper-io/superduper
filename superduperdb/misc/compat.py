"""
Functions from later standard libraries not available in Python 3.8
"""

from functools import lru_cache
import typing as t

__all__ = ('cache',)


# Implements functools.cache from Python 3.9
def cache(user_function: t.Callable, /) -> t.Any:
    return lru_cache(maxsize=None)(user_function)
