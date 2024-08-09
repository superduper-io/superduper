from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(
    ['transformers', '4.29.1'], ['datasets', '2.18.0'], ['torch']
)

from .model import LLM, TextClassificationPipeline

__version__ = "0.3.0"

__all__ = ('TextClassificationPipeline', 'LLM')
