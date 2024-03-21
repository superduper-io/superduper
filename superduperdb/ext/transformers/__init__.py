from superduperdb.misc.annotations import requires_packages

requires_packages(
    ['transformers', '4.29.1'],
    ['datasets', '2.18.0'],
    ['torch']
)

from .model import LLM, TextClassificationPipeline
from .training import LLMTrainer

__all__ = ('TextClassificationPipeline', 'LLM', 'LLMTrainer')
