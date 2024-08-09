from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(['cohere', '4.40'])

from .model import CohereEmbed, CohereGenerate

__version__ = '0.3.0'

__all__ = 'CohereEmbed', 'CohereGenerate'
