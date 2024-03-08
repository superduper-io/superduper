from superduperdb.misc.annotations import requires_packages

requires_packages(
    ['transformers', '4.29.1'],
    ['datasets', '2.18.0'],
)

from .model import TextClassificationPipeline

__all__ = ('TextClassificationPipeline',)
