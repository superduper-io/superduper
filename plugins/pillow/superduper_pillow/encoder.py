import io

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
from superduper.base.datatype import (
    BaseDataType,
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

    def _decode_data(self, item):
        """Decode the data.

        :param item: The data to decode.
        """
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


pil_image = PILImage()
pil_image_hybrid = PILImageHybrid()
