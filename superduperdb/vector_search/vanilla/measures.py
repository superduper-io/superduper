import typing as t

import numpy
from numpy import ndarray


ArrayLike = t.TypeVar('ArrayLike', list, tuple, range, str, bytes)


def l2(x, y) -> ndarray:
    return numpy.array([-numpy.linalg.norm(x - y, axis=1)])


def dot(x, y) -> ndarray:
    return numpy.dot(x, y.T)


def css(x: ndarray, y: ndarray) -> ndarray:
    x = x / numpy.linalg.norm(x, axis=1)[:, None]
    y = y / numpy.linalg.norm(y, axis=1)[:, None]
    return dot(x, y)
