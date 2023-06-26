import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
import io
import typing as t

from superduperdb.core.encoder import Encoder


def encode_pil_image(x: t.Any) -> bytes:
    buffer = io.BytesIO()
    x.save(buffer, format='png')
    return buffer.getvalue()


def decode_pil_image(bytes: bytes) -> PIL.Image.Image:
    return PIL.Image.open(io.BytesIO(bytes))


pil_image = Encoder(
    'pil_image',
    encoder=encode_pil_image,
    decoder=decode_pil_image,
)
