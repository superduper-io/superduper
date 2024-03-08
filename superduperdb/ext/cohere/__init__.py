from superduperdb.misc.annotations import requires_packages

requires_packages(['cohere', '4.40'])

from .model import CohereEmbed, CohereGenerate

__all__ = 'CohereEmbed', 'CohereGenerate'
