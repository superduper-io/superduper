# ruff: noqa: E402
from .base import config, configs, logger
from .base.superduper import superduper

ICON = '🔮'
CFG = configs.CFG
ROOT = configs.ROOT

logging = logger.Logging

__version__ = '0.1.1'

from superduperdb.backends import ibis, mongodb

from .base.document import Document
from .components.dataset import Dataset
from .components.datatype import DataType, Encoder
from .components.listener import Listener
from .components.metric import Metric
from .components.model import Model, ObjectModel
from .components.schema import Schema
from .components.vector_index import VectorIndex, vector

__all__ = (
    'CFG',
    'ICON',
    'ROOT',
    'config',
    'logging',
    'superduper',
    'DataType',
    'Encoder',
    'Document',
    'ObjectModel',
    'Model',
    'Listener',
    'VectorIndex',
    'vector',
    'Dataset',
    'Metric',
    'Schema',
    'Serializer',
    'mongodb',
    'ibis',
)
