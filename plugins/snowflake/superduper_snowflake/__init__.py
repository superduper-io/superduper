from .data_backend import SnowflakeDataBackend as DataBackend
from .secrets import check_secret_updates, raise_if_secrets_pending, secrets_not_ready
from .vector_search import SnowflakeVectorSearcher as VectorSearcher

__version__ = "0.7.0"

__all__ = [
    "VectorSearcher",
    "DataBackend",
    "check_secret_updates",
    "raise_if_secrets_pending",
    "secrets_not_ready",
]
