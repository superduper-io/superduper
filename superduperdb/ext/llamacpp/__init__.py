from superduperdb.misc.annotations import requires_packages

_, requirements = requires_packages(['llama_cpp_python', '0.2.39'])

from .model import LlamaCpp, LlamaCppEmbedding

__all__ = ['LlamaCpp', 'LlamaCppEmbedding']
