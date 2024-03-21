from superduperdb.misc.annotations import requires_packages

requires_packages(
    ['transformers', '4.29.1'],
    ['datasets', '2.18.0'],
    ['torch']
)

from .llm import LLM
from .llm_training import LLMTrainer
from .model import TextClassificationPipeline

__all__ = ('TextClassificationPipeline', 'LLM', 'LLMTrainer')
