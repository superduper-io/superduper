import numpy


def l2(x, y):
    return numpy.array([-numpy.linalg.norm(x - y, axis=1)])


def dot(x, y):
    return numpy.dot(x, y.T)


def css(x, y):
    x = x / numpy.linalg.norm(x, axis=1)[:, None]
    y = y / numpy.linalg.norm(y, axis=1)[:, None]
    return dot(x, y)
