import typing as t

from superduper.components.component import Component


class Metric(Component):
    """Metric base object used to evaluate performance on a dataset.

    These objects are callable and are applied row-wise to the data, and averaged.

    :param object: Callable or an Artifact to be applied to the data.
    """

    type_id: t.ClassVar[str] = 'metric'

    object: t.Callable

    def __call__(self, x: t.Sequence[int], y: t.Sequence[int]) -> bool:
        """Call the metric object on the x and y data.

        :param x: First sequence of data.
        :param y: Second sequence of data.
        """
        return self.object(x, y)
