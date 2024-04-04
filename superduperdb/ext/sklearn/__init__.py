from superduperdb.misc.annotations import requires_packages

requires_packages(
    ['sklearn', '1.2.2', None, 'scikit-learn'],
)

from .model import Estimator, SklearnTrainer

__all__ = 'Estimator', 'SklearnTrainer'
