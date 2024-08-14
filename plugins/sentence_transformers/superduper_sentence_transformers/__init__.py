from superduper.misc.annotations import requires_packages

from .model import SentenceTransformer

__version__ = "0.0.2"

__all__ = ('SentenceTransformer',)

_, requirements = requires_packages(['sentence-transformers', '2.2.2', None])
