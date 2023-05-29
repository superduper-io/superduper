from superduperdb.core.base import Component


class Metric(Component):
    variety = 'metric'

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
