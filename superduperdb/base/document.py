import importlib
import re
import typing as t

from bson.objectid import ObjectId

from superduperdb import logging
from superduperdb.base.code import Code
from superduperdb.base.leaf import Leaf, find_leaf_cls
from superduperdb.base.serializable import Serializable
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    _ENCODABLES,
    DataType,
    Encodable,
    _BaseEncodable,
)
from superduperdb.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.schema import Schema


ContentType = t.Union[t.Dict, Encodable]
ItemType = t.Union[t.Dict[str, t.Any], Encodable, ObjectId]

_LEAF_TYPES = {
    'component': Component,
    'serializable': Serializable,
    'remote_code': Code,
}
_LEAF_TYPES.update(_ENCODABLES)
_OUTPUTS_KEY = '_outputs'


class Document(MongoStyleDict):
    """A wrapper around an instance of dict or a Encodable.

    The document data is used to dump that resource to a mix of json-able content, ids and `bytes`

    """

    _DEFAULT_ID_KEY: str = '_id'

    def encode(
        self,
        schema: t.Optional['Schema'] = None,
        leaf_types_to_keep: t.Sequence[t.Type] = (),
    ) -> t.Dict:
        """Make a copy of the content with all the Leaves encoded.

        :param schema: The schema to encode with.
        :param leaf_types_to_keep: The types of leaves to keep.
        """
        if schema is not None:
            return _encode_with_schema(dict(self), schema)
        return _encode(dict(self), leaf_types_to_keep)

    def get_leaves(self, *leaf_types: str):
        """Get all the leaves in the document.

        :param *leaf_types: The types of leaves to get.
        """
        keys, leaves = _find_leaves(self, *leaf_types)
        return dict(zip(keys, leaves))

    @property
    def variables(self) -> t.List[str]:
        """Return a list of variables in the object."""
        from superduperdb.base.serializable import _find_variables

        return _find_variables(self)

    def set_variables(self, db: 'Datalayer', **kwargs) -> 'Document':
        """Set free variables of self.

        :param db: The datalayer to use.
        """
        from superduperdb.base.serializable import _replace_variables

        content = _replace_variables(
            self, db, **kwargs
        )  # replace variables with values
        return Document(**content)

    @staticmethod
    def decode(r: t.Dict, db: t.Optional['Datalayer'] = None) -> t.Any:
        """Decode the object from a encoded data.

        :param r: Encoded data.
        :param db: Datalayer instance.
        """
        cache = {}
        if '_leaves' in r:
            r['_leaves'] = _build_leaves(r['_leaves'], db=db)
            cache = {sub['id']: sub for sub in r['_leaves']}
        decoded = _decode(dict(r), db=db, leaf_cache=cache)
        if isinstance(decoded, dict):
            return Document(decoded)
        return decoded

    def __repr__(self) -> str:
        return f'Document({repr(dict(self))})'

    def unpack(self, db=None, leaves_to_keep: t.Sequence = ()) -> t.Any:
        """Returns the content, but with any encodables replaced by their contents.

        :param db: The datalayer to use.
        :param leaves_to_keep: The types of leaves to keep.
        """
        out = _unpack(self, db=db, leaves_to_keep=leaves_to_keep)
        if '_base' in out:
            out = out['_base']
        return out


def _find_leaves(r: t.Any, *leaf_types: str, pop: bool = False):
    if isinstance(r, dict):
        keys = []
        leaves = []
        for k, v in r.items():
            sub_keys, sub_leaves = _find_leaves(v, *leaf_types)
            leaves.extend(sub_leaves)
            keys.extend([(f'{k}.{sub_key}' if sub_key else k) for sub_key in sub_keys])
        return keys, leaves
    if isinstance(r, list) or isinstance(r, tuple):
        keys = []
        leaves = []
        for i, x in enumerate(r):
            sub_keys, sub_leaves = _find_leaves(x, *leaf_types)
            leaves.extend(sub_leaves)
            keys.extend([(f'{k}.{i}' if k else f'{i}') for k in sub_keys])
        return keys, leaves
    if leaf_types:
        leaf_clses = [
            _LEAF_TYPES.get(leaf_type) or find_leaf_cls(leaf_type)
            for leaf_type in leaf_types
        ]
        if isinstance(r, tuple(leaf_clses)):
            return [''], [r]
        else:
            return [], []
    else:
        if isinstance(r, Leaf):
            return [''], [r]
        else:
            return [], []


def _decode(
    r: t.Any,
    db: t.Optional[t.Any] = None,
    leaf_cache: t.Optional[t.Dict] = None,
) -> t.Any:
    if isinstance(r, dict) and '_content' in r:
        leaf_type = r['_content']['leaf_type']
        leaf_cls = _LEAF_TYPES.get(leaf_type) or find_leaf_cls(leaf_type)
        return leaf_cls.decode(r, db=db)
    elif isinstance(r, str) and re.match(
        '^_(artifact|lazy_artifact|component|encodable)\/', r
    ):
        assert leaf_cache is not None, 'Leaf cache must be provided to decode'
        return leaf_cache[r]
    elif isinstance(r, list):
        return [_decode(x, db=db, leaf_cache=leaf_cache) for x in r]
    elif isinstance(r, dict):
        return {k: _decode(v, db=db, leaf_cache=leaf_cache) for k, v in r.items()}
    else:
        return r


def _encode_with_references(r: t.Any, references: t.Dict):
    if isinstance(r, dict):
        for k, v in r.items():
            if isinstance(v, Leaf):
                r[k] = f'$_{v.leaf_type}s/{v.unique_id}'
                references[f'_{v.leaf_type}s'][v.unique_id] = v
            else:
                _encode_with_references(r[k], references=references)
    if isinstance(r, list):
        for i, x in enumerate(r):
            if isinstance(x, Leaf):
                r[i] = f'$_{x.leaf_type}s/{x.unique_id}'
                references[f'_{x.leaf_type}'][x.unique_id] = x
            else:
                _encode_with_references(x, references=references)


def _encode(
    r: t.Any,
    leaf_types_to_keep: t.Sequence[t.Type] = (),
) -> t.Any:
    if isinstance(r, dict):
        out = {}
        for k, v in r.items():
            out[k] = _encode(v, leaf_types_to_keep)
        return out
    if isinstance(r, list) or isinstance(r, tuple):
        out = []  # type: ignore[assignment]
        for x in r:
            out.append(_encode(x, leaf_types_to_keep))
        return out
    # ruff: noqa: E501
    if isinstance(r, Leaf) and not isinstance(r, leaf_types_to_keep):  # type: ignore[arg-type]
        return r.encode(leaf_types_to_keep=leaf_types_to_keep)
    return r


def _encode_with_schema(r: t.Any, schema: 'Schema') -> t.Any:
    if isinstance(r, dict):
        out = {}
        for k, v in r.items():
            if isinstance(schema.fields.get(k, None), DataType):
                assert isinstance(schema.fields[k], DataType)
                out[k] = schema.fields[k].encode_data(v)
            else:
                tmp = _encode_with_schema(v, schema)
                out[k] = tmp
        return out
    if isinstance(r, Leaf):
        return r.encode()
    return r


def _unpack(item: t.Any, db=None, leaves_to_keep: t.Sequence = ()) -> t.Any:
    if isinstance(item, _BaseEncodable) and not isinstance(item, leaves_to_keep):  # type: ignore[arg-type]
        return item.unpack(db=db)
    elif isinstance(item, dict):
        return {
            k: _unpack(v, db=db, leaves_to_keep=leaves_to_keep) for k, v in item.items()
        }
    elif isinstance(item, list):
        return [_unpack(x, db=db, leaves_to_keep=leaves_to_keep) for x in item]
    else:
        return item


class NotBuiltError(Exception):
    """Exception for when a leaf is not built.

    :param key: The key that was not built.
    """

    def __init__(self, *args, key, **kwargs):
        super().__init__(*args, **kwargs)
        self.key = key


def _encode_flattened(r, leaf_types_to_keep, leaf_cache):
    if isinstance(r, Leaf):
        encoded = r.encode()['_content']
        leaf_cache[encoded['id']] = encoded
        return encoded['id']
    if isinstance(r, (list, tuple)):
        return [_encode_flattened(x, leaf_types_to_keep, leaf_cache) for x in r]
    if isinstance(r, dict):
        return {
            k: _encode_flattened(v, leaf_types_to_keep, leaf_cache)
            for k, v in r.items()
        }
    return r


def _fetch_cache_keys(r, cache, used):
    if isinstance(r, str) and r.startswith('_'):
        try:
            out = cache[r]
            used.append(r)
            return out
        except KeyError:
            raise NotBuiltError(
                f"Cache key {r} not found in cache: available: {cache.keys()}",
                key=r,
            )
    elif isinstance(r, dict):
        return {k: _fetch_cache_keys(v, cache, used) for k, v in r.items() if k != 'id'}
    elif isinstance(r, list):
        return [_fetch_cache_keys(x, cache, used) for x in r]
    return r


def _build_leaf(leaf_record, cache):
    module = importlib.import_module(leaf_record['module'])
    cls = getattr(module, leaf_record['cls'])
    r = cls.handle_integration(leaf_record['dict'])
    built = cls.build(r)
    cache[built.id] = built
    return built


def _build_leaves(leaf_records, db=None):
    cache = {}
    if db is not None:
        cache.update(
            {f'_component/datatype/{c}': db.datatypes[c] for c in db.datatypes}
        )
    default_keys = []
    if db is not None:
        default_keys = [f'_component/datatype/{c}' for c in db.datatypes]
    used = []
    while True:
        missing_keys = []
        built = []
        for i, r in enumerate(leaf_records):
            try:
                r = _fetch_cache_keys(r, cache, used)
            except NotBuiltError as e:
                logging.warn(str(e))
                missing_keys.append(e.key)
                continue
            built.append(i)
            _build_leaf(r, cache)
        assert built, f"Infinite loop in leaf building; missing keys: {missing_keys}"
        leaf_records = [r for i, r in enumerate(leaf_records) if i not in built]
        if not leaf_records:
            break
    exit_leaves = [k for k in cache.keys() if k not in used and k not in default_keys]
    return {k: v for k, v in cache.items() if k not in default_keys}, exit_leaves


def _deep_flat_encode(r, cache):
    if isinstance(r, dict):
        return {k: _deep_flat_encode(v, cache) for k, v in r.items()}
    if isinstance(r, list):
        return [_deep_flat_encode(x, cache) for x in r]
    if isinstance(r, Leaf):
        return r._deep_flat_encode(cache)
    return r
