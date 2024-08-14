from superduper.misc.annotations import requires_packages

from .training import TorchTrainer

_, requirements = requires_packages(['torch', '2.0.0'])

from .encoder import tensor
from .model import TorchModel, torchmodel

__version__ = "0.0.2"

__all__ = ('TorchModel', 'TorchTrainer', 'tensor', 'torchmodel')
