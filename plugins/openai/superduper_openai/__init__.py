from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(['openai', '1.1.2', None], ['httpx'])

from .model import OpenAIChatCompletion, OpenAIEmbedding

__version__ = '0.3.0'

__all__ = 'OpenAIChatCompletion', 'OpenAIEmbedding'
