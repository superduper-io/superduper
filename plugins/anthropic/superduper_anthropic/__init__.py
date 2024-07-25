from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(['anthropic', '0.25.0'])

from .model import AnthropicCompletions

__all__ = ('AnthropicCompletions',)
