import io
import typing as t

import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin

from superduperdb.components.datatype import DataType

BLANK_IMAGE = PIL.Image.new('RGB', (600, 600), (255, 255, 255))


def encode_pil_image(x, info: t.Optional[t.Dict] = None):
    buffer = io.BytesIO()
    x.save(buffer, 'png')
    return buffer.getvalue()


class DecoderPILImage:
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
