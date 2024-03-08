from superduperdb.misc.annotations import requires_packages

requires_packages(['openai', '1.1.2', None], ['httpx'])

from .model import OpenAIChatCompletion, OpenAIEmbedding

__all__ = 'OpenAIChatCompletion', 'OpenAIEmbedding'
