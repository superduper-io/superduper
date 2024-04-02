from superduperdb.ext.vllm.model import VllmAPI, VllmModel
from superduperdb.misc.annotations import requires_packages

__all__ = ["VllmAPI", "VllmModel"]

_, requirements = requires_packages(['vllm', None, None], ['ray', None, None], warn=True)
