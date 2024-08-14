from superduper.misc.annotations import requires_packages

_, requirements = requires_packages(['PIL', '10.2.0', None, 'pillow'])

from .encoder import pil_image

__version__ = "0.0.2"

__all__ = ['pil_image']
