# ruff: noqa: E402
from .base import config, config_settings, configs, logger
from .base.superduper import superduper

CFG = configs.CFG
ROOT = config_settings.ROOT

logging = logger.Logging

from importlib import metadata

try:
    __version__ = metadata.version('superduper-framework')
except metadata.PackageNotFoundError:
    # when developers do `pip install -e .`
    __version__ = "dev"


from .base.datatype import BaseDataType, dill_serializer, pickle_serializer
from .base.document import Document
from .base.schema import Schema
from .components.application import Application
from .components.component import Component, trigger
from .components.cron_job import CronJob, FunctionCronJob
from .components.dataset import Dataset
from .components.listener import Listener
from .components.metric import Metric
from .components.model import (
    APIBaseModel,
    Model,
    ObjectModel,
    QueryModel,
    Trainer,
    Validation,
)
from .components.plugin import Plugin
from .components.streamlit import Streamlit
from .components.table import Table
from .components.template import Template
from .components.vector_index import VectorIndex

REQUIRES = [
    'superduper=={}'.format(__version__),
]

__all__ = (
    'CFG',
    'ROOT',
    'config',
    'logging',
    'superduper',
    'BaseDataType',
    'Document',
    'ObjectModel',
    'QueryModel',
    'Validation',
    'APIBaseModel',
    'Model',
    'CronJob',
    'FunctionCronJob',
    'Trainer',
    'Listener',
    'VectorIndex',
    'Dataset',
    'Metric',
    'Plugin',
    'Schema',
    'Table',
    'Application',
    'Template',
    'Application',
    'Component',
    'trigger',
    'pickle_serializer',
    'dill_serializer',
    'Streamlit',
)
