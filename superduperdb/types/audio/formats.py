import io
from scipy.io.wavfile import read, write


class WavAudio:
    """
    >>> with open('tests/material/data/test.wav', 'rb') as f: bs = f.read()
    >>> fs, arr = WavAudio.decode(bs)  # doctest: +ELLIPSIS
    >>> fs
    22050
    >>> nbs = WavAudio.encode((fs, arr))
    >>> nbs == bs
    True
    """

    def __init__(self, types=()):
        self.types = types

    @staticmethod
    def decode(x):
        return read(io.BytesIO(x))

    @staticmethod
    def encode(x):
        rate, data = x
        byte_io = io.BytesIO(bytes())
        write(byte_io, rate, data)
        return byte_io.getvalue()
