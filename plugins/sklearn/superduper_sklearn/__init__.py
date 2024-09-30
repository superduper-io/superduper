from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(
    ['sklearn', '1.2.2', None, 'scikit-learn'],
)

from .model import Estimator, SklearnTrainer

__version__ = "0.0.5"

__all__ = 'Estimator', 'SklearnTrainer'
