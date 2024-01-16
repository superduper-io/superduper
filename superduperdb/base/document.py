import typing as t

import bson
from bson.objectid import ObjectId

from superduperdb import CFG
from superduperdb.base.config import BytesEncoding
from superduperdb.components.encoder import Encodable, Encoder
from superduperdb.misc.files import get_file_from_uri
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.components.schema import Schema


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

    def encode(
        self,
        schema: t.Optional['Schema'] = None,
        bytes_encoding: t.Optional[BytesEncoding] = None,
    ) -> t.Any:
        """Make a copy of the content with all the Encodables encoded"""
        bytes_encoding = bytes_encoding or CFG.bytes_encoding
        if schema is not None:
            return _encode_with_schema(self.content, schema, bytes_encoding)
        return _encode(self.content, bytes_encoding)

    @property
    def variables(self) -> t.List[str]:
        from superduperdb.base.serializable import _find_variables

        return _find_variables(self.content)

    def set_variables(self, db, **kwargs) -> 'Document':
        from superduperdb.base.serializable import _replace_variables

        content = _replace_variables(
            self.content, db, **kwargs
        )  # replace variables with values
        return Document(content)

    def outputs(self, key: str, model: str, version: t.Optional[int] = None) -> t.Any:
        """
        Get document ouputs on ``key`` from ``model``

        :param key: Document key to get outputs from.
        :param model: Model name to get outputs from.
        """
        r = MongoStyleDict(self.unpack())
        if version is not None:
            document = r[f'{_OUTPUTS_KEY}.{key}.{model}.{version}']
        else:
            tmp = r[f'{_OUTPUTS_KEY}.{key}.{model}']
            version = max(list(tmp.keys()))
            return tmp[version]
        return document

    @staticmethod
    def decode(
        r: t.Dict, encoders: t.Dict, bytes_encoding: t.Optional[BytesEncoding] = None
    ) -> t.Any:
        bytes_encoding = bytes_encoding or CFG.bytes_encoding

        if isinstance(r, Document):
            return Document(_decode(r, encoders, bytes_encoding))
        elif isinstance(r, dict):
            return _decode(r, encoders, bytes_encoding)
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


def _decode(
    r: t.Dict, encoders: t.Dict, bytes_encoding: t.Optional[BytesEncoding] = None
) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding
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
                r[k] = encoders[k].decode(r[k], bytes_encoding).x
            else:
                r[k] = _decode(r[k], encoders, bytes_encoding)
    return r


def _encode(r: t.Any, bytes_encoding: t.Optional[BytesEncoding] = None) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding

    if isinstance(r, dict):
        return {k: _encode(v, bytes_encoding) for k, v in r.items()}
    if isinstance(r, Encodable):
        return r.encode(bytes_encoding=bytes_encoding)
    return r


def _encode_with_schema(
    r: t.Any, schema: 'Schema', bytes_encoding: t.Optional[BytesEncoding] = None
) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding
    if isinstance(r, dict):
        out = {
            k: schema.fields[k].encode(v, wrap=False)  # type: ignore[call-arg]
            if isinstance(schema.fields[k], Encoder)
            else _encode_with_schema(v, schema, bytes_encoding)
            for k, v in r.items()
        }
        return out
    if isinstance(r, Encodable):
        return r.encode(bytes_encoding=bytes_encoding)
    return r


def _unpack(item: t.Any) -> t.Any:
    if isinstance(item, Encodable):
        if CFG.hybrid_storage and not item.encoder.load_hybrid and item.x is None:
            return get_file_from_uri(item.uri)
        return item.x
    elif isinstance(item, dict):
        return {k: _unpack(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unpack(x) for x in item]
    else:
        return item
