from superduperdb.misc.annotations import requires_packages

requires_packages(['anthropic', '0.12.0'])

from .model import AnthropicCompletions

__all__ = ('AnthropicCompletions',)
