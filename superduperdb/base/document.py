import re
import typing as t

from bson.objectid import ObjectId

from superduperdb.base.code import Code
from superduperdb.base.leaf import Leaf, _import_item
from superduperdb.base.variables import Variable
from superduperdb.components.component import Component
from superduperdb.components.datatype import (
    _ENCODABLES,
    Blob,
    Encodable,
    FileItem,
    _BaseEncodable,
)
from superduperdb.components.schema import SCHEMA_KEY, Schema, get_schema
from superduperdb.misc.reference import parse_reference
from superduperdb.misc.special_dicts import MongoStyleDict, SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


ContentType = t.Union[t.Dict, Encodable]
ItemType = t.Union[t.Dict[str, t.Any], Encodable, ObjectId]
LeafMetaType = t.Type['Leaf']

_LEAF_TYPES = {
    'component': Component,
    'leaf': Leaf,
    'remote_code': Code,
}
_LEAF_TYPES.update(_ENCODABLES)
_OUTPUTS_KEY = '_outputs'


class Document(MongoStyleDict):
    """A wrapper around an instance of dict or a Encodable.

    The document data is used to dump that resource to
    a mix of json-able content, ids and `bytes`

    :param args: *args for `dict`
    :param schema: The schema to use.
    :param db: The datalayer to use.
    :param kwargs: **kwargs for `dict`
    """

    _DEFAULT_ID_KEY: str = '_id'

    def __init__(
        self,
        *args,
        schema: t.Optional['Schema'] = None,
        db: t.Optional['Datalayer'] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.db = db
        self.schema = schema

    def encode(
        self,
        schema: t.Optional[t.Union['Schema', str]] = None,
        leaves_to_keep: t.Sequence = (),
    ) -> SuperDuperFlatEncode:
        """Encode the document to a format that can be used in a database.

        After encoding everything is a vanilla dictionary (JSON + bytes).
        (Even a model, or artifact etc..)

        :param schema: The schema to use.
        :param leaves_to_keep: The types of leaves to keep.
        """
        builds: t.Dict[str, dict] = self.get('_leaves', {})
        blobs: t.Dict[str, bytes] = self.get('_blobs', {})
        files: t.Dict[str, str] = self.get('_files', {})

        # Get schema from database.
        schema = self.schema or schema
        schema = get_schema(self.db, schema) if schema else None
        out = dict(self)
        if schema is not None:
            out = schema.encode_data(
                out, builds, blobs, files, leaves_to_keep=leaves_to_keep
            )

        out = _deep_flat_encode(
            out,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
        )
        # TODO - don't need to save in one document
        # can return encoded, builds, files, blobs
        out.update({'_leaves': builds, '_files': files, '_blobs': blobs})
        out = SuperDuperFlatEncode(out)
        return out

    @classmethod
    def decode(
        cls,
        r,
        schema: t.Optional['Schema'] = None,
        db: t.Optional['Datalayer'] = None,
        getters: t.Optional[t.Dict[str, t.Callable]] = None,
    ):
        """Converts any dictionary into a Document or a Leaf.

        :param r: The encoded data.
        :param schema: The schema to use.
        :param db: The datalayer to use.
        """
        schema = schema or r.get(SCHEMA_KEY)
        schema = get_schema(db, schema)

        if schema is not None:
            schema.init()
            r = schema.decode_data(r)

        builds = r.get('_leaves', {})

        # Important: Leaf.identifier is used as the key, but must be set if not present.
        for k in builds:
            if isinstance(builds[k], dict) and (
                '_path' in builds[k] or '_object' in builds[k]
            ):
                if 'identifier' not in builds[k]:
                    builds[k]['identifier'] = k

        getters = getters or {}
        if r.get('_blobs'):

            def _get_from_local_artifacts(x):
                return r['_blobs'].get(x, x)

            getters['blob'] = _get_from_local_artifacts

        if r.get('_files'):

            def _get_from_local_files(x):
                return r['_files'].get(x.split(':')[-1], x)

            getters['file'] = _get_from_local_files

        if db is not None:
            getters['component'] = lambda x: _get_component(db, x)

        r = _deep_flat_decode(
            {k: v for k, v in r.items() if k not in ('_leaves', '_blobs', '_files')},
            builds=builds,
            db=db,
            getters=getters,
        )

        if isinstance(r, dict):
            return Document(r, schema=schema)
        else:
            return r

    @property
    def variables(self) -> t.List[str]:
        """Return a list of variables in the object."""
        from superduperdb.base.variables import _find_variables

        return _find_variables(self)

    def set_variables(self, **kwargs) -> 'Document':
        """Set free variables of self.

        :param db: The datalayer to use.
        :param kwargs: The vales to set the variables to `_replace_variables`.
        """
        from superduperdb.base.variables import _replace_variables

        content = _replace_variables(self, **kwargs)
        return Document(**content)

    def __repr__(self) -> str:
        return f'Document({repr(dict(self))})'

    def unpack(self, leaves_to_keep: t.Sequence = ()) -> t.Any:
        """Returns the content, but with any encodables replaced by their contents.

        :param leaves_to_keep: The types of leaves to keep.
        """
        out = _unpack(self, leaves_to_keep=leaves_to_keep)
        if '_base' in out:
            out = out['_base']
        return out


class QueryUpdateDocument(Document):
    """A document that is used to update a document in a database.

    This document is used to update a document in a database.
    It is a subclass of Document.

    :param args: *args for `dict`
    :param schema: The schema to use.
    :param db: The datalayer to use.
    :param kwargs: **kwargs for `dict`
    """

    @classmethod
    def from_document(cls, document):
        """Create a QueryUpdateDocument from a document.

        :param document: The document to create the QueryUpdateDocument from.
        """
        r = dict(document)
        r = r.get('$set', r)
        return QueryUpdateDocument(r)

    @staticmethod
    def _create_metadata_update(update, original=None):
        # Update leaves
        if original is None:
            original = update
        if not isinstance(original, SuperDuperFlatEncode):
            return {'$set': update}

        for mk in ('_leaves', '_files', '_blobs'):
            m = original.pop(mk, {})
            for k, v in m.items():
                update[f'{mk}.{k}'] = v

        update = {'$set': update}
        return update

    def encode(
        self,
        original: t.Any = None,
        schema: t.Optional['Schema'] = None,
        leaves_to_keep: t.Sequence = (),
    ) -> t.Dict:
        """Encode the document to a format that can be used in a database.

        :param original: The original document.
        :param schema: The schema to use.
        :param leaves_to_keep: The types of leaves to keep.
        """
        r = dict(self)
        r = r.get('$set', r)
        update = super().encode(schema=schema, leaves_to_keep=leaves_to_keep)
        return self._create_metadata_update(update, original=original)


def _unpack(item: t.Any, db=None, leaves_to_keep: t.Sequence = ()) -> t.Any:
    if isinstance(item, _BaseEncodable) and not any(
        [isinstance(item, leaf) for leaf in leaves_to_keep]
    ):
        return item.unpack()
    elif isinstance(item, dict):
        return {k: _unpack(v, leaves_to_keep=leaves_to_keep) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unpack(x, leaves_to_keep=leaves_to_keep) for x in item]
    else:
        return item


def _deep_flat_encode(
    r,
    builds,
    blobs,
    files,
    leaves_to_keep=(),
):
    if isinstance(r, dict):
        tmp = {}
        for k in list(r):
            tmp[k] = _deep_flat_encode(
                r[k],
                builds,
                blobs=blobs,
                files=files,
                leaves_to_keep=leaves_to_keep,
            )
        return tmp

    if isinstance(r, (list, tuple)):
        return type(r)(
            [
                _deep_flat_encode(
                    x,
                    builds,
                    blobs=blobs,
                    files=files,
                    leaves_to_keep=leaves_to_keep,
                )
                for x in r
            ]
        )

    if isinstance(r, Document):
        return _deep_flat_encode(
            r,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
        )

    if isinstance(r, Blob):
        blobs[r.identifier] = r.bytes
        return '&:blob:' + r.identifier

    if isinstance(r, FileItem):
        files[r.identifier] = r.path
        return '&:file:' + r.reference

    if isinstance(r, Leaf):
        if isinstance(r, leaves_to_keep):
            builds[r.identifier] = r
            return '?' + r.identifier

        r = r.dict()
        r = _deep_flat_encode(
            r,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
        )

        identifier = r.pop('identifier')
        builds[identifier] = r
        return f'?{identifier}'

    if isinstance(r, Variable):
        return r.key

    return r


def _get_leaf_from_cache(k, builds, getters, db: t.Optional['Datalayer'] = None):
    if reference := parse_reference(f'?{k}'):
        if reference.name in getters:
            return getters[reference.name](reference.path)

    if isinstance(builds[k], Leaf):
        leaf = builds[k]
        if not leaf.db:
            leaf.db = db
        return leaf
    leaf = _deep_flat_decode(builds[k], builds, getters=getters, db=db)
    builds[k] = leaf
    if isinstance(leaf, Leaf):
        if not leaf.db:
            leaf.db = db
    return leaf


def _deep_flat_decode(
    r, builds, getters: t.Optional[dict], db: t.Optional['Datalayer'] = None
):
    if isinstance(r, Leaf):
        return r
    if isinstance(r, (list, tuple)):
        return type(r)(
            [_deep_flat_decode(x, builds, getters=getters, db=db) for x in r]
        )
    if isinstance(r, dict) and '_path' in r:
        parts = r['_path'].split('/')
        cls = parts[-1]
        module = '.'.join(parts[:-1])
        dict_ = {k: v for k, v in r.items() if k != '_path'}
        dict_ = _deep_flat_decode(dict_, builds, getters=getters, db=db)
        instance = _import_item(cls=cls, module=module, dict=dict_, db=db)
        return instance
    if isinstance(r, dict) and '_object' in r:
        dict_ = {k: v for k, v in r.items() if k != '_object'}
        dict_ = _deep_flat_decode(dict_, builds, getters=getters, db=db)
        object = _deep_flat_decode(
            builds[r['_object'][1:]], builds, getters=getters, db=db
        )
        instance = _import_item(object=object.unpack(), dict=dict_, db=db)
        return instance
    if isinstance(r, dict):
        literals = r.get('_literals', [])
        return {
            k: _deep_flat_decode(v, builds, getters=getters, db=db)
            if k not in literals
            else v
            for k, v in r.items()
        }
    if isinstance(r, str) and r.startswith('?'):
        return _get_leaf_from_cache(r[1:], builds, getters=getters, db=db)
    if isinstance(r, str) and r.startswith('&'):
        assert getters is not None
        reference = parse_reference(r)
        if reference.name in getters:
            return getters[reference.name](reference.path)
        return r
    if isinstance(r, str) and (vars := re.findall(r'^<var:(.*?)>$', r)):
        return Variable(vars[0])

    return r


def _get_component(db, path):
    parts = path.split(':')
    if len(parts) == 1:
        return db.load(uuid=parts[0])
    if len(parts) == 2:
        return db.load(type_id=parts[0], identifier=parts[1])
    if len(parts) == 3:
        if not parts[2].isnumeric():
            return db.load(uuid=parts[2])
        return db.load(type_id=parts[0], identifier=parts[1], version=parts[2])
    raise ValueError(f'Invalid component reference: {path}')


# TODO: How about the lazy loading of the artifact?
# Should this function return a callback function for getting the artifact?
def _get_artifact(db, path):
    return db.artifact_store.get_bytes(path)
