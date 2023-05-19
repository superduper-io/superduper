import PIL.Image
import PIL.JpegImagePlugin
import PIL.PngImagePlugin
import io


class Image:
    """
    >>> with open('img/black.png', 'rb') as f: bs = f.read()
    >>> im = Image.decode(bs)
    >>> bs = Image.encode(im)
    """

    types = (PIL.JpegImagePlugin.JpegImageFile, PIL.PngImagePlugin.PngImageFile)

    @staticmethod
    def encode(x):
        buffer = io.BytesIO()
        x.save(buffer, format='png')
        return buffer.getvalue()

    @staticmethod
    def decode(bytes_):
        return PIL.Image.open(io.BytesIO(bytes_))
