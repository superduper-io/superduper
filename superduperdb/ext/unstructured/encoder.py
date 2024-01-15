import dataclasses as dc
import pickle
import typing as t

from unstructured.documents.elements import Element
from unstructured.partition.auto import partition
from unstructured.partition.html import partition_html

from superduperdb.components.encoder import Encoder


def link2elements(link, unstructure_kwargs):
    """
    Convert a link to a list of elements
    Use unstructured to parse the link
    param link: str, file path or url
    param unstructure_kwargs: kwargs for unstructured
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
    def encoder(x: t.Union[str, t.List[Element]]):
        if isinstance(x, str):
            elements = link2elements(x, unstructure_kwargs)
        elif isinstance(x, list) and isinstance(x[0], Element):
            elements = x
        else:
            raise TypeError(f"Cannot encode {type(x)}, must be str or list of Element")
        return pickle.dumps(elements)

    return encoder


def create_decoder():
    def decoder(b: bytes):
        try:
            return pickle.loads(b)
        except Exception as e:
            # TODO: A compatibility issue with unstructured when using uri
            # When superduperdb download uri and pass the bytes to the decoder,
            # the file type message is lost, unstructured cannot  automatically parse.
            raise ValueError(
                "Cannot parse the bytes from uri data, please use encoder(x=uri)"
            ) from e

    return decoder


@dc.dataclass
class UnstructuredEncoder(Encoder):
    unstructure_kwargs: t.Dict[str, t.Any] = dc.field(default_factory=dict)

    def __post_init__(self):
        self.encoder = create_encoder(self.unstructure_kwargs)
        self.decoder = create_decoder()
        super().__post_init__()


unstructured_encoder = UnstructuredEncoder(
    identifier="unstructured",
)
