from superduper.misc.annotations import requires_packages

from superduper_vllm.model import VllmAPI, VllmModel

__version__ = "0.0.2"

__all__ = ["VllmAPI", "VllmModel"]

_, requirements = requires_packages(
    ['vllm', None, None], ['ray', None, None], warn=True
)
