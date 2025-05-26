from .data_backend import MongoDBDataBackend as DataBackend
from .vector_search import MongoAtlasVectorSearcher as VectorSearcher

__version__ = "0.7.0"

__all__ = [
    "DataBackend",
    "VectorSearcher",
]
