from .cache import RedisCache as Cache
from .data_backend import RedisDataBackend as DataBackend

__version__ = '0.6.0'

__all__ = ['Cache', 'DataBackend']
