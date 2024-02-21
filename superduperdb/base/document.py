import dataclasses as dc
import os
import typing as t

from bson.objectid import ObjectId

from superduperdb import CFG
from superduperdb.backends.base.artifact import _construct_file_id_from_uri
from superduperdb.base.config import BytesEncoding
from superduperdb.base.leaf import Leaf
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component
from superduperdb.components.datatype import DataType, Encodable
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.schema import Schema


ContentType = t.Union[t.Dict, Encodable]
ItemType = t.Union[t.Dict[str, t.Any], Encodable, ObjectId]

_OUTPUTS_KEY: str = '_outputs'
_LEAF_TYPES = {
    'component': Component,
    'encodable': Encodable,
    'serializable': Serializable,
}


class Document(MongoStyleDict):
    """
    A wrapper around an instance of dict or a Encodable which may be used to dump
    that resource to a mix of json-able content, ids and `bytes`

    :param content: The content to wrap
    """

    _DEFAULT_ID_KEY: str = '_id'

    def encode(
        self,
        schema: t.Optional['Schema'] = None,
        leaf_types_to_keep: t.Sequence[t.Type] = (),
        bytes_encoding: t.Optional[BytesEncoding] = None,
    ) -> t.Tuple[dict, t.List[Leaf]]:
        """Make a copy of the content with all the Leaves encoded"""
        bytes_encoding = bytes_encoding or CFG.bytes_encoding
        if schema is not None:
            return _encode_with_schema(dict(self), schema, bytes_encoding)
        return _encode(dict(self), bytes_encoding, leaf_types_to_keep)

    def get_leaves(self, leaf_type: t.Optional[str] = None):
        keys, leaves = _find_leaves(self, leaf_type)
        return dict(zip(keys, leaves))

    @property
    def variables(self) -> t.List[str]:
        from superduperdb.base.serializable import _find_variables

        return _find_variables(self)

    def set_variables(self, db, **kwargs) -> 'Document':
        from superduperdb.base.serializable import _replace_variables

        content = _replace_variables(
            self, db, **kwargs
        )  # replace variables with values
        return Document(**content)

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
        r: t.Dict,
        db: t.Optional['Datalayer'] = None,
        bytes_encoding: t.Optional[BytesEncoding] = None,
        reference: bool = False,
    ) -> t.Any:
        bytes_encoding = bytes_encoding or CFG.bytes_encoding
        decoded = _decode(
            dict(r), db=db, bytes_encoding=bytes_encoding, reference=reference
        )
        if isinstance(decoded, dict):
            return Document(decoded)
        return decoded

    def __repr__(self) -> str:
        return f'Document({repr(dict(self))})'

    def unpack(self, db=None) -> t.Any:
        """Returns the content, but with any encodables replaced by their contents"""
        out = _unpack(self, db=db)
        if '_base' in out:
            out = out['_base']
        return out


def _find_leaves(r: t.Any, leaf_type: t.Optional[str] = None, pop: bool = False):
    if isinstance(r, dict):
        keys = []
        leaves = []
        for k, v in r.items():
            sub_keys, sub_leaves = _find_leaves(v, leaf_type)
            leaves.extend(sub_leaves)
            keys.extend([(f'{k}.{sub_key}' if sub_key else k) for sub_key in sub_keys])
        return keys, leaves
    if isinstance(r, list) or isinstance(r, tuple):
        keys = []
        leaves = []
        for i, x in enumerate(r):
            sub_keys, sub_leaves = _find_leaves(x, leaf_type)
            leaves.extend(sub_leaves)
            keys.extend([(f'{k}.{i}' if k else f'{i}') for k in sub_keys])
        return keys, leaves
    if leaf_type:
        if isinstance(r, _LEAF_TYPES[leaf_type]):
            return [''], [r]
        else:
            return [], []
    else:
        if isinstance(r, Leaf):
            return [''], [r]
        else:
            return [], []


def _decode(
    r: t.Dict,
    db: t.Optional[t.Any] = None,
    bytes_encoding: t.Optional[BytesEncoding] = None,
    reference: bool = False,
) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding
    if isinstance(r, dict) and '_content' in r:
        return _LEAF_TYPES[r['_content']['leaf_type']].decode(
            r, db=db, reference=reference
        )
    elif isinstance(r, list):
        return [
            _decode(x, db=db, bytes_encoding=bytes_encoding, reference=reference)
            for x in r
        ]
    elif isinstance(r, dict):
        return {
            k: _decode(v, db=db, bytes_encoding=bytes_encoding, reference=reference)
            for k, v in r.items()
        }
    else:
        return r


@dc.dataclass
class Reference(Serializable):
    identifier: str
    leaf_type: str
    path: t.Optional[str] = None
    db: t.Optional['Datalayer'] = None


def _encode_with_references(r: t.Any, references: t.Dict):
    if isinstance(r, dict):
        for k, v in r.items():
            if isinstance(v, Leaf):
                r[k] = f'${v.leaf_type}/{v.unique_id}'
                references[v.leaf_type][v.unique_id] = v
            else:
                _encode_with_references(r[k], references=references)
    if isinstance(r, list):
        for i, x in enumerate(r):
            if isinstance(x, Leaf):
                ref = Reference(x.unique_id, leaf_type=x.leaf_type)
                r[i] = ref
                references[x.leaf_type][x.unique_id] = x
            else:
                _encode_with_references(x, references=references)


def _encode(
    r: t.Any,
    bytes_encoding: t.Optional[BytesEncoding] = None,
    leaf_types_to_keep: t.Sequence[t.Type] = (),
) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding
    if isinstance(r, dict):
        out = {}
        for k, v in r.items():
            out[k] = _encode(v, bytes_encoding, leaf_types_to_keep)
        return out
    if isinstance(r, list) or isinstance(r, tuple):
        out = []  # type: ignore[assignment]
        for x in r:
            out.append(_encode(x, bytes_encoding, leaf_types_to_keep))
        return out
    # ruff: noqa: E501
    if isinstance(r, Leaf) and not isinstance(r, leaf_types_to_keep):  # type: ignore[arg-type]
        # TODO: (not leaf_types_to_keep or isinstance(r, leaf_types_to_keep)) ?
        return r.encode(
            bytes_encoding=bytes_encoding, leaf_types_to_keep=leaf_types_to_keep
        )
    return r


def _encode_with_schema(
    r: t.Any,
    schema: 'Schema',
    bytes_encoding: t.Optional[BytesEncoding] = None,
) -> t.Any:
    bytes_encoding = bytes_encoding or CFG.bytes_encoding
    if isinstance(r, dict):
        out = {}
        for k, v in r.items():
            if isinstance(schema.fields.get(k, None), DataType):
                assert isinstance(schema.fields[k], DataType)
                out[k] = schema.fields[k].encoder(v)  # type: ignore[union-attr]
            else:
                tmp = _encode_with_schema(v, schema, bytes_encoding)
                out[k] = tmp
        return out
    if isinstance(r, Leaf):
        return r.encode(bytes_encoding=bytes_encoding)
    return r


def _unpack(item: t.Any, db=None) -> t.Any:
    if isinstance(item, Encodable):
        # TODO move logic into Encodable
        if item.reference:
            file_id = _construct_file_id_from_uri(item.uri)
            if item.datatype.directory:
                file_id = os.path.join(item.datatype.directory, file_id)
            return file_id
        return item.unpack(db=db)
    elif isinstance(item, dict):
        return {k: _unpack(v, db=db) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unpack(x, db=db) for x in item]
    else:
        return item
