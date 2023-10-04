import typing as t

import bson
from bson.objectid import ObjectId

from superduperdb.container.encoder import Encodable, Encoder
from superduperdb.container.schema import Schema

ContentType = t.Union[t.Dict, Encodable]
ItemType = t.Union[t.Dict[str, t.Any], Encodable, ObjectId]

_OUTPUTS_KEY: str = '_outputs'


class Document:
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of JSONable and `bytes`

    :param content: The content to wrap
    """

    _DEFAULT_ID_KEY: str = '_id'

    content: ContentType

    def __init__(self, content: ContentType):
        self.content = content

    def dump_bson(self) -> bytes:
        """Dump this document into BSON and encode as bytes"""
        return bson.encode(self.encode())

    def encode(self, schema: t.Optional[Schema] = None) -> t.Any:
        """Make a copy of the content with all the Encodables encoded"""
        if schema is not None:
            return _encode_with_schema(self.content, schema)
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
        assert isinstance(self.content, dict)
        return self.content[item]

    def __setitem__(self, key: str, value: ItemType):
        assert isinstance(self.content, dict)
        self.content[key] = value

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
    document: t.Dict = bson.decode(content)
    return Document(Document.decode(document, encoders=encoders))


def load_bsons(content: t.ByteString, encoders: t.Dict) -> t.List[Document]:
    """Load a list of Documents from bson-encoded content

    :param content: the content to decode
    :param encoders: a dict of encoders
    """
    d = t.cast(t.Dict, bson.decode(content))
    documents = d['docs']
    return [Document(Document.decode(r, encoders=encoders)) for r in documents]


def _decode(r: t.Dict, encoders: t.Dict) -> t.Any:
    if isinstance(r, dict) and '_content' in r:
        encoder = encoders[r['_content']['encoder']]
        try:
            return encoder.decode(r['_content']['bytes'])
        except KeyError:
            if 'uri' in r['_content']:
                return Encodable(uri=r['_content']['uri'], encoder=encoder)
            return r
    elif isinstance(r, list):
        return [_decode(x, encoders) for x in r]
    elif isinstance(r, dict):
        for k in r:
            if k in encoders:
                r[k] = encoders[k].decode(r[k]).x
            else:
                r[k] = _decode(r[k], encoders)
    return r


def _encode(r: t.Any) -> t.Any:
    if isinstance(r, dict):
        return {k: _encode(v) for k, v in r.items()}
    if isinstance(r, Encodable):
        return r.encode()
    if isinstance(r, (bool, int, str, ObjectId)):
        return r
    return r


def _encode_with_schema(r: t.Any, schema: Schema) -> t.Any:
    if isinstance(r, dict):
        return {
            k: schema.fields[k].encode(v, wrap=False)  # type: ignore[call-arg]
            if isinstance(schema.fields[k], Encoder)
            else _encode_with_schema(v, schema)
            for k, v in r.items()
        }
    if isinstance(r, Encodable):
        return r.encode()
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
