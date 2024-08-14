from superduper.ext.vllm.model import VllmAPI, VllmModel
from superduper.misc.annotations import requires_packages

__version__ = "0.0.1"

__all__ = ["VllmAPI", "VllmModel"]

_, requirements = requires_packages(
    ['vllm', None, None], ['ray', None, None], warn=True
)
