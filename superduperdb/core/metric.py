from superduperdb.core.base import Component


class Metric(Component):
    """
    Metric base object with which to evaluate performance on a data-set.
    These objects are ``callable`` and are applied row-wise to the data, and averaged.
    """
    variety = 'metric'

    def __call__(self, x, y):
        raise NotImplementedError
