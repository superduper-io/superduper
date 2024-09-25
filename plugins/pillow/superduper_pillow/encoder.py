import io
import typing as t

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
from superduper.components.datatype import DataType, DataTypeFactory
from superduper.misc.annotations import component

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer

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
    """Decoder to convert `bytes` back into a `PIL.Image` class # noqa.

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
    """Create a `DataType` for an image.

    :param identifier: The identifier for the data type.
    :param encodable: The encodable type.
    :param media_type: The media type.
    :param db: The datalayer instance.
    """
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


class PilDataTypeFactory(DataTypeFactory):
    """A factory for pil image # noqa."""

    @staticmethod
    def check(data: t.Any) -> bool:
        """Check if the data is a image.

        It's used for registering the auto schema.
        :param data: The data to check.
        """
        return isinstance(data, PIL.Image.Image)

    @staticmethod
    def create(data: t.Any) -> DataType:
        """Create a pil_image datatype.

        It's used for registering the auto schema.
        :param data: The image data.
        """
        return pil_image
