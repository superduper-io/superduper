import io
import typing as t

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
from superduper.components.datatype import (
    BaseDataType,
    DataTypeFactory,
    _Artifact,
    _Encodable,
)

BLANK_IMAGE = PIL.Image.new('RGB', (600, 600), (255, 255, 255))


class _PILImageMixin(BaseDataType):
    """Mixin class for pil-image encodings."""

    def _encode_data(self, item):
        buffer = io.BytesIO()
        item.save(buffer, 'png')
        return buffer.getvalue()

    def decode_data(self, item):
        try:
            return PIL.Image.open(io.BytesIO(item))
        except Exception as e:
            if self.handle_exceptions:
                return BLANK_IMAGE
            else:
                raise e


class PILImage(_Encodable, _PILImageMixin, BaseDataType):
    """PIL Images saved in databackend."""


class PILImageHybrid(_Artifact, _PILImageMixin, BaseDataType):
    """PIL Images saved as artifacts."""


pil_image = PILImage('pil_image')
pil_image_hybrid = PILImage('pil_image_hybrid')


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
    def create(data: t.Any) -> BaseDataType:
        """Create a pil_image datatype.

        It's used for registering the auto schema.
        :param data: The image data.
        """
        return pil_image
