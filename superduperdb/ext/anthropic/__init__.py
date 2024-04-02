from superduperdb.misc.annotations import requires_packages

_, requirements = requires_packages(['anthropic', '0.12.0'])

from .model import AnthropicCompletions

__all__ = ('AnthropicCompletions',)
