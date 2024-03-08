from superduperdb.misc.annotations import requires_packages

requires_packages(['torch', '2.0.0'])

from .encoder import tensor
from .model import TorchModel, TorchTrainer, torchmodel

__all__ = ('TorchModel', 'TorchTrainer', 'tensor', 'torchmodel')
