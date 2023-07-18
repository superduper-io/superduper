from superduperdb.core.encoder import Encodable
import bson
import typing as t

ContentType = t.Union[t.Dict, Encodable]


class Document:
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of jsonable content or `bytes`
    """

    _DEFAULT_ID_KEY = '_id'

    def __init__(self, content: t.Dict):
        self.content = content

    def __hash__(self):
        return super().__hash__()

    def dump_bson(self):
        return bson.encode(self.encode())

    # ruff: noqa: E501
    @staticmethod
    def load_bson(content, encoders):
        return Document(Document.decode(bson.decode(content), encoders=encoders))  # type: ignore[arg-type]

    @staticmethod
    def dump_bsons(documents):
        return bytes(bson.encode({'docs': [d.encode() for d in documents]}))

    @staticmethod
    def load_bsons(content: bytearray, encoders: t.Dict):
        documents = bson.decode(content)['docs']  #  type: ignore[arg-type]
        return [Document(Document.decode(r, encoders=encoders)) for r in documents]

    def _encode(self, r: t.Any):
        if isinstance(r, dict):
            return {k: self._encode(v) for k, v in r.items()}
        elif isinstance(r, Encodable):
            return r.encode()
        return r

    def encode(self):
        return self._encode(self.content)

    @property
    def id(
        self,
    ):
        return self.content[self._DEFAULT_ID_KEY]

    @classmethod
    def decode(cls, r: t.Dict, encoders: t.Dict):
        if isinstance(r, Document):
            return Document(cls._decode(r, encoders))
        elif isinstance(r, dict):
            return cls._decode(r, encoders)
        raise NotImplementedError(f'type {type(r)} is not supported')

    @classmethod
    def _decode(cls, r: t.Dict, encoders: t.Dict):
        if isinstance(r, dict) and '_content' in r:
            type = encoders[r['_content']['encoder']]
            try:
                return type.decode(r['_content']['bytes'])
            except KeyError:
                return r
        elif isinstance(r, list):
            return [cls._decode(x, encoders) for x in r]
        elif isinstance(r, dict):
            for k in r:
                r[k] = cls._decode(r[k], encoders)
        return r

    def __repr__(self):
        return f'Document({self.content.__repr__()})'

    def __getitem__(self, item: str):
        return self.content[item]

    def __setitem__(self, key: str, value: t.Any):
        self.content[key] = value

    @classmethod
    def _unpack_datavars(cls, item: t.Any):
        if isinstance(item, Encodable):
            return item.x
        elif isinstance(item, dict):
            return {k: cls._unpack_datavars(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [cls._unpack_datavars(x) for x in item]
        else:
            return item

    def unpack(self):
        return self._unpack_datavars(self.content)
