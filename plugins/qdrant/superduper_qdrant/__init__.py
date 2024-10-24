from superduper.misc.annotations import requires_packages

from .qdrant import QdrantVectorSearcher as VectorSearcher

_, requirements = requires_packages(['qdrant-client', '1.10.0', '2'])

__version__ = "0.0.1"

__all__ = ['VectorSearcher']
