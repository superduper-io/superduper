import io
import typing as t

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin

from superduperdb.components.datatype import DataType
from superduperdb.misc.annotations import component

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer

BLANK_IMAGE = PIL.Image.new('RGB', (600, 600), (255, 255, 255))


def encode_pil_image(x, info: t.Optional[t.Dict] = None):
    buffer = io.BytesIO()
    x.save(buffer, 'png')
    return buffer.getvalue()


class DecoderPILImage:
    """
    Decoder to convert `bytes` back into a `PIL.Image` class

    :param handle_exceptions: return a blank image if failure
    """

    def __init__(self, handle_exceptions: bool = True):
        self.handle_exceptions = handle_exceptions

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        try:
            return PIL.Image.open(io.BytesIO(bytes))
        except Exception as e:
            if self.handle_exceptions:
                return BLANK_IMAGE
            else:
                raise e


decode_pil_image = DecoderPILImage()


@component(
    {'name': 'identifier', 'type': 'str'},
    {'name': 'media_type', 'type': 'str', 'default': 'image/png'},
)
def image_type(
    identifier: str,
    encodable: str = 'lazy_artifact',
    media_type: str = 'image/png',
    db: t.Optional['Datalayer'] = None,
):
    return DataType(
        identifier=identifier,
        encoder=encode_pil_image,
        decoder=decode_pil_image,
        encodable=encodable,
        media_type=media_type,
    )


pil_image = image_type(
    identifier='pil_image',
    encodable='encodable',
)

pil_image_hybrid = image_type(
    identifier='pil_image_hybrid',
    encodable='artifact',
)
