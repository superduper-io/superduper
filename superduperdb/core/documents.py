from superduperdb.core.encoder import Encodable
from superduperdb.misc.uri_cache import Cached
import typing as t

ContentType = t.Union[t.Dict, Encodable]


class Document(Cached[ContentType]):
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of jsonable content or `bytes`
    """

    def __hash__(self):
        return super().__hash__()

    def _encode(self, r: t.Any):
        if isinstance(r, dict):
            return {k: self._encode(v) for k, v in r.items()}
        elif isinstance(r, Encodable):
            return r.encode()
        return r

    def __repr__(self):
        return f'Document(uri={self.uri}, _content={self._content}'

    def encode(self):
        return self._encode(self.content)

    @classmethod
    def decode(cls, r: t.Dict, types: t.Dict):
        if isinstance(r, Document):
            return Document(cls._decode(r, types))
        elif isinstance(r, dict):
            return cls._decode(r, types)
        raise NotImplementedError(f'type {type(r)} is not supported')

    @classmethod
    def _decode(cls, r: t.Dict, types: t.Dict):
        if isinstance(r, dict) and '_content' in r:
            type = types[r['_content']['type']]
            try:
                return type.decode(r['_content']['bytes'])
            except KeyError:
                return r
        elif isinstance(r, list):
            return [cls._decode(x, types) for x in r]
        elif isinstance(r, dict):
            for k in r:
                r[k] = cls._decode(r[k], types)
        return r

    def __getitem__(self, item: str):
        assert isinstance(self.content, dict)
        return self.content[item]

    def __setitem__(self, key: str, value: t.Any):
        assert isinstance(self.content, dict)
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
