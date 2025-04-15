from .data_backend import SnowflakeDataBackend as DataBackend
from .secrets import check_secret_updates
from .vector_search import SnowflakeVectorSearcher as VectorSearcher

__version__ = "0.6.0"

__all__ = [
    "VectorSearcher",
    "DataBackend",
    "check_secret_updates",
]
