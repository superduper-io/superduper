from superduper.misc.annotations import requires_packages

from .lance import LanceVectorSearcher as VectorSearcher

_, requirements = requires_packages(['pylance', '0.6.1', '0.8.14'])

__version__ = "0.0.6"

__all__ = ['VectorSearcher']
