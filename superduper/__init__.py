# ruff: noqa: E402
from .base import config, config_settings, configs, logger
from .base.superduper import superduper

ICON = '🔮'

CFG = configs.CFG
ROOT = config_settings.ROOT

logging = logger.Logging

import toml
import os

def get_version():
    path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    with open(path, 'r', encoding='utf-8') as file:
        pyproject = toml.load(file)
    return pyproject['project']['version']

__version__ = get_version()


from .base.decorators import code
from .base.document import Document
from .base.leaf import imported, imported_value
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
from .components.streamlit import Streamlit
from .components.table import Table
from .components.template import QueryTemplate, Template
from .components.vector_index import VectorIndex, vector

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
    'Template',
    'QueryTemplate',
    'Application',
    'Component',
    'pickle_serializer',
    'dill_serializer',
    'templates',
    'imported',
    'imported_value',
    'Streamlit',
)
