import typing as t

import bson
from bson.objectid import ObjectId

import superduperdb as s
from superduperdb.container.encoder import Encodable

ContentType = t.Union[t.Dict, Encodable]
ItemType = t.Union[t.Dict[str, t.Any], Encodable, ObjectId]

_OUTPUTS_KEY: str = '_outputs'


class Document:
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of JSONable and `bytes`
    """

    _DEFAULT_ID_KEY: str = '_id'

    content: ContentType

    def __init__(self, content: ContentType):
        self.content = content

    def dump_bson(self) -> bytes:
        """Dump this document into BSON and encode as bytes"""
        return bson.encode(self.encode())

    def encode(self) -> t.Any:
        """Make a copy of the content with all the Encodables encoded"""
        return _encode(self.content)

    def outputs(self, key: str, model: str) -> t.Any:
        """
        Get document ouputs on ``key`` from ``model``

        :param key: Document key to get outputs from.
        :param model: Model name to get outputs from.
        """
        document = self.unpack()[_OUTPUTS_KEY][key][model]
        return document

    @staticmethod
    def decode(r: t.Dict, encoders: t.Dict) -> t.Any:
        if isinstance(r, Document):
            return Document(_decode(r, encoders))
        elif isinstance(r, dict):
            return _decode(r, encoders)
        raise NotImplementedError(f'type {type(r)} is not supported')

    def __repr__(self) -> str:
        return f'Document({repr(self.content)})'

    def __getitem__(self, item: str) -> ItemType:
        return self.content[item]  # type: ignore[index]

    def __setitem__(self, key: str, value: ItemType):
        self.content[key] = value  # type: ignore[index]

    def unpack(self) -> t.Any:
        """Returns the content, but with any encodables replacecs by their contents"""
        return _unpack(self.content)


def dump_bsons(documents: t.Sequence[Document]) -> bytes:
    """Dump a sequence of documents into BSON and encode as bytes

    :param documents: the sequence of documents to dump
    """
    return bytes(bson.encode({'docs': [d.encode() for d in documents]}))


def load_bson(content: t.ByteString, encoders: t.Dict[str, t.Any]) -> Document:
    """Load a Document from bson-encoded content

    :param content: the content to decode
    :param encoders: a dict of encoders
    """
    document = bson.decode(content)  # type: ignore[arg-type, var-annotated]
    return Document(Document.decode(document, encoders=encoders))


def load_bsons(content: t.ByteString, encoders: t.Dict) -> t.List[Document]:
    """Load a list of Documents from bson-encoded content

    :param content: the content to decode
    :param encoders: a dict of encoders
    """
    documents = bson.decode(content)['docs']  #  type: ignore[arg-type, index]
    return [Document(Document.decode(r, encoders=encoders)) for r in documents]


# ruff: noqa: E501
def _decode(r: t.Dict, encoders: t.Dict) -> t.Any:
    if isinstance(r, dict) and '_content' in r:
        encoder = encoders[r['_content']['encoder']]
        try:
            return encoder.decode(r['_content']['bytes'])
        except KeyError:
            if 'uri' in r['_content']:
                return Encodable(uri=r['_content']['uri'], encoder=encoder)  # type: ignore[call-arg]
            return r
    elif isinstance(r, list):
        return [_decode(x, encoders) for x in r]
    elif isinstance(r, dict):
        for k in r:
            r[k] = _decode(r[k], encoders)
    return r


def _encode(r: t.Any) -> t.Any:
    if isinstance(r, dict):
        return {k: _encode(v) for k, v in r.items()}
    if isinstance(r, Encodable):
        return r.encode()
    if isinstance(r, (bool, int, str, bson.ObjectId)):
        return r
    s.log.info(f'Unexpected type {type(r)} in Document.encode')
    return r


def _unpack(item: t.Any) -> t.Any:
    if isinstance(item, Encodable):
        return item.x
    elif isinstance(item, dict):
        return {k: _unpack(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unpack(x) for x in item]
    else:
        return item
