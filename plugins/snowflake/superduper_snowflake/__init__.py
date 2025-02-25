from .vector_search import SnowflakeVectorSearcher as VectorSearcher
from .data_backend import SnowflakeDataBackend as DataBackend
from .secrets import check_secret_updates

__version__ = "0.5.21"

__all__ = [
    "VectorSearcher",
    "DataBackend",
    "check_secret_updates",
]
