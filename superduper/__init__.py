# ruff: noqa: E402
import os

from .base import config, configs, config_settings, logger
from .base.superduper import superduper

ICON = '🔮'

CFG = configs.CFG
ROOT = config_settings.ROOT

logging = logger.Logging

__version__ = '0.3.0'

# from superduper.backends import ibis, mongodb

from .base.decorators import code
from .base.document import Document
from .components.application import Application
from .components.component import Component
from .components.dataset import Dataset
from .components.datatype import DataType, dill_serializer, pickle_serializer
from .components.listener import Listener
from .components.metric import Metric
from .components.model import (
    Model,
    ObjectModel,
    QueryModel,
    Validation,
    model,
)
from .components.plugin import Plugin
from .components.schema import Schema
from .components.table import Table
from .components.template import Template
from .components.vector_index import VectorIndex, vector
from .misc.annotations import requires_packages

REQUIRES = [
    'superduper=={}'.format(__version__),
]

__all__ = (
    'CFG',
    'ICON',
    'ROOT',
    'config',
    'logging',
    'superduper',
    'DataType',
    'Document',
    'code',
    'ObjectModel',
    'QueryModel',
    'Validation',
    'Model',
    'model',
    'Listener',
    'VectorIndex',
    'vector',
    'Dataset',
    'Metric',
    'Plugin',
    'Schema',
    'Table',
    'Application',
    'mongodb',
    'ibis',
    'Template',
    'Application',
    'Component',
    'requires_packages',
    'pickle_serializer',
    'dill_serializer',
)
