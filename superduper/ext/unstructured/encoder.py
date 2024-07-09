import pickle
import typing as t

from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html

from superduper.components.datatype import DataType


def link2elements(link, unstructure_kwargs):
    """Convert a link to a list of elements.

    Use unstructured to parse the link
    :param link: str, file path or url
    :param unstructure_kwargs: kwargs for unstructured
    """
    if link.startswith("file://"):
        link = link[7:]

    if link.startswith("http"):
        # Special handling is required for http
        elements = partition_html(url=link, **unstructure_kwargs)
    else:
        elements = partition(link, **unstructure_kwargs)
    return elements


def create_encoder(unstructure_kwargs):
    """Create an encoder for unstructured data.

    :param unstructure_kwargs: kwargs for unstructured
    """

    def encoder(x: t.Union[str, t.List[Element]], info: t.Optional[t.Dict] = None):
        if isinstance(x, str):
            elements = link2elements(x, unstructure_kwargs)
        elif isinstance(x, list) and isinstance(x[0], Element):
            elements = x
        else:
            raise TypeError(f"Cannot encode {type(x)}, must be str or list of Element")
        return pickle.dumps(elements)

    return encoder


def create_decoder():
    """Create a decoder for unstructured data."""

    def decoder(b: bytes, info: t.Optional[t.Dict] = None):
        try:
            return pickle.loads(b)
        except Exception as e:
            # TODO: A compatibility issue with unstructured when using uri
            # When superduper download uri and pass the bytes to the decoder,
            # the file type message is lost, unstructured cannot automatically parse.
            raise ValueError(
                "Cannot parse the bytes from uri data, please use encoder(x=uri)"
            ) from e

    return decoder


unstructured_encoder = DataType(
    identifier="unstructured",
    encoder=create_encoder({}),
    decoder=create_decoder(),
)


def create_unstructured_encoder(identifier, **unstructure_kwargs):
    """Create an unstructured encoder with the given identifier and unstructure kwargs.

    :param identifier: The identifier to use.
    :param *unstructure_kwargs: The unstructure kwargs to use.
    """
    assert (
        isinstance(identifier, str) and identifier != "unstructured"
    ), 'identifier must be a string and not "unstructured"'
    return DataType(
        identifier=identifier,
        encoder=create_encoder(unstructure_kwargs),
        decoder=create_decoder(),
    )
