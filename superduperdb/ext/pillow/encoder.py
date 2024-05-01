import io
import typing as t

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin

from superduperdb.components.datatype import DataType
from superduperdb.misc.annotations import component

BLANK_IMAGE = PIL.Image.new('RGB', (600, 600), (255, 255, 255))


def encode_pil_image(x, info: t.Optional[t.Dict] = None):
    """Encode a `PIL.Image` to bytes.

    :param x: The image to encode.
    :param info: Additional information.
    """
    buffer = io.BytesIO()
    x.save(buffer, 'png')
    return buffer.getvalue()


class DecoderPILImage:
    """Decoder to convert `bytes` back into a `PIL.Image` class.

    :param handle_exceptions: return a blank image if failure
    """

    def __init__(self, handle_exceptions: bool = True):
        self.handle_exceptions = handle_exceptions

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        """Decode a `PIL.Image` from bytes.

        :param bytes: The bytes to decode.
        :param info: Additional information.
        """
        try:
            return PIL.Image.open(io.BytesIO(bytes))
        except Exception as e:
            if self.handle_exceptions:
                return BLANK_IMAGE
            else:
                raise e


decode_pil_image = DecoderPILImage()


pil_image = DataType(
    'pil_image',
    encoder=encode_pil_image,
    decoder=decode_pil_image,
)

pil_image_hybrid = DataType(
    'pil_image_hybrid',
    encoder=encode_pil_image,
    decoder=decode_pil_image,
    encodable='artifact',
)

pil_image_hybrid_png = DataType(
    'pil_image_hybrid_png',
    encoder=encode_pil_image,
    decoder=decode_pil_image,
    encodable='artifact',
    media_type='image/png',
)

pil_image_hybrid_jpeg = DataType(
    'pil_image_hybrid_jpeg',
    encoder=encode_pil_image,
    decoder=decode_pil_image,
    encodable='artifact',
    media_type='image/jpeg',
)


@component(
    {'name': 'identifier', 'type': 'str'},
    {'name': 'media_type', 'type': 'str', 'default': 'image/png'},
)
def image_type(
    identifier: str, encodable: str = 'lazy_artifact', media_type: str = 'image/png'
):
    """Create a `DataType` for an image.

    :param identifier: The identifier for the data type.
    :param encodable: The encodable type.
    :param media_type: The media type.
    """
    return DataType(
        identifier,
        encoder=encode_pil_image,
        decoder=decode_pil_image,
        encodable=encodable,
        media_type=media_type,
    )
