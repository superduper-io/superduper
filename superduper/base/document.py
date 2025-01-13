import inspect
import typing as t
from collections import defaultdict

from superduper import CFG, logging
from superduper.base.constant import (
    KEY_BLOBS,
    KEY_BUILDS,
    KEY_FILES,
    KEY_SCHEMA,
)
from superduper.base.leaf import Leaf, import_item
from superduper.base.variables import _replace_variables
from superduper.components.component import Component
from superduper.components.datatype import BaseDataType, Blob, File
from superduper.components.schema import Schema, get_schema
from superduper.misc.reference import parse_reference
from superduper.misc.special_dicts import MongoStyleDict, SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


LeafMetaType = t.Type['Leaf']

_VERSION_LIMIT = 1000


class Getters:
    """A class to manage getters for decoding documents.

    We have will have a list of getters for each type of reference.

    For example:
    - blob: [load_from_blobs, load_from_db]
    - file: [load_from_files, load_from_db]

    :param getters: A dictionary of getters.
    """

    def __init__(self, getters=None):
        self._getters = defaultdict(list)
        for k, v in (getters or {}).items():
            self.add_getter(k, v)

    def add_getter(self, name: str, getter: t.Callable):
        """Add a getter for a reference type."""
        self._getters[name].append(getter)
        # if name == 'blob':
        #     self._getters[name].append(_build_blob_getter(getter))
        # else:
        #     self._getters[name].append(getter)

    def run(self, name, data):
        """Run the getters one by one until one returns a value."""
        if name not in self._getters:
            return data
        for getter in self._getters[name]:
            out = getter(data)
            if out is not None:
                return out
        return data


def _diff(r1, r2, d):
    # TODO deal with the case when the implementing class is different

    for k in r1:
        if not isinstance(r1[k], type(r2[k])):
            d[k] = r2[k]
            continue

        if isinstance(r1[k], dict):
            if r1[k].keys() != r2[k].keys():
                d[k] = r2[k]
                continue

        if isinstance(r1[k], dict):
            subdiff = {}
            _diff(r1[k], r2[k], {})
            if subdiff:
                d[k] = subdiff
            continue

        if isinstance(r1[k], Leaf):
            r1k = r1[k].dict(metadata=False)

            if r2[k] is None:
                d[k] = None
                continue

            r2k = r2[k].dict(metadata=False)

            if set(r1k.keys()) != set(r2k.keys()):
                d[k] = r2[k]
                continue

            if 'uuid' in r1k:
                del r1k['uuid']
            if 'uuid' in r2k:
                del r2k['uuid']
            dd = {}
            _diff(r1k, r2k, dd)
            if dd:
                d[k] = r2[k]
            continue

        if r1[k] != r2[k]:
            d[k] = r2[k]


def _update(r, s):
    # TODO - how to deal with unordered sets?
    """
    Update a dictionary with another dictionary, also nested.

    >>> r = {'a': 1, 'b': {'c': 2, 'd': 3}}
    >>> s = {'b': {'c': 4}}
    >>> _update(r, s)
    {'a': 1, 'b': {'c': 4, 'd': 3}}
    """
    for k in s:
        if isinstance(s[k], dict):
            r[k] = _update(r.get(k, {}), s[k])
        else:
            r[k] = s[k]
    return r


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

    def map(self, fn, condition):
        def _map(r):
            if isinstance(r, dict):
                out = {}
                for k, v in r.items():
                    out[k] = _map(v)
                return out
            if isinstance(r, (list, tuple, set)):
                out = []
                for x in r:
                    out.append(_map(x))
                return type(r)(out)
            if condition(r):
                return fn(r)
            return r

        return Document(_map(self), schema=self.schema)

    def diff(self, other: 'Document'):
        """Get a `Document` with the difference to `other` inside.

        :param other: Other `Document`.
        """
        out: t.Dict = {}
        _diff(self, other, out)
        return Document(out, schema=self.schema)

    def update(self, other: t.Union['Document', dict]):
        """Update document with values from other."""
        schema = self.schema or Schema('tmp', fields={})

        if isinstance(other, Document) and other.schema:
            assert other.schema is not None
            schema = schema.update(other.schema)
        return Document(_update(dict(self), dict(other)), schema=schema)

    def encode(
        self,
        schema: t.Optional[t.Union['Schema', str]] = None,
        leaves_to_keep: t.Sequence = (),
        metadata: bool = True,
        defaults: bool = True,
        keep_schema: bool = True,
    ) -> SuperDuperFlatEncode:
        """Encode the document to a format that can be used in a database.

        After encoding everything is a vanilla dictionary (JSON + bytes).
        (Even a model, or artifact etc..)

        :param schema: The schema to use.
        :param leaves_to_keep: The types of leaves to keep.
        """
        builds: t.Dict[str, dict] = self.get(KEY_BUILDS, {})
        blobs: t.Dict[str, bytes] = self.get(KEY_BLOBS, {})
        files: t.Dict[str, str] = self.get(KEY_FILES, {})

        # Get schema from database.
        schema = self.schema or schema
        schema = get_schema(self.db, schema) if schema else None
        out = dict(self)

        if schema is not None:
            out = schema.encode_data(
                out, builds, blobs, files, leaves_to_keep=leaves_to_keep
            )

        if not keep_schema:
            del out['_schema']

        out = _deep_flat_encode(
            out,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
            metadata=metadata,
            defaults=defaults,
        )
        # TODO - don't need to save in one document
        # can return encoded, builds, files, blobs
        out.update({KEY_BUILDS: builds, KEY_FILES: files, KEY_BLOBS: blobs})
        out = SuperDuperFlatEncode(out)
        return out

    def __getitem__(self, key: str) -> t.Any:
        if not key.startswith(CFG.output_prefix) or '.' in key:
            return super().__getitem__(key)

        import re

        if re.match(f'{CFG.output_prefix}[^_]{1,}__[a-z0-9]{10,}', key):
            return super().__getitem__(key)

        key = next(k for k in self.keys() if k.startswith(key))
        return super().__getitem__(key)

    @classmethod
    def decode(
        cls,
        r,
        schema: t.Optional['Schema'] = None,
        db: t.Optional['Datalayer'] = None,
        getters: t.Union[Getters, t.Dict[str, t.Callable], None] = None,
    ):
        """Converts any dictionary into a Document or a Leaf.

        :param r: The encoded data.
        :param schema: The schema to use.
        :param db: The datalayer to use.
        """
        if '_variables' in r:
            variables = {**r['_variables'], 'output_prefix': CFG.output_prefix}
            r = _replace_variables(
                {k: v for k, v in r.items() if k != '_variables'}, **variables
            )
        schema = schema or r.get(KEY_SCHEMA)
        schema = get_schema(db, schema)

        builds = r.get(KEY_BUILDS, {})

        # TODO is this the right place for this?
        # Important: Leaf.identifier or Component.type_id:Component.identifier are
        # are used as the key, but must be set if not present.
        for k in builds:
            if isinstance(builds[k], dict) and (
                '_path' in builds[k] or '_object' in builds[k]
            ):
                if 'identifier' not in builds[k]:
                    # Component.type_id:Component.identifier
                    if ":" in k:
                        _, identifier = k.split(':', 1)
                    # Leaf.identifier
                    else:
                        identifier = k
                    builds[k]['identifier'] = identifier

        if not isinstance(getters, Getters):
            getters = Getters(getters)
        assert isinstance(getters, Getters)

        # Prioritize using the local artifact storage getter,
        # and then use the DB read getter.
        if r.get(KEY_BLOBS):
            getters.add_getter(
                'blob', lambda x: Blob(identifier=x, bytes=r[KEY_BLOBS].get(x))
            )

        def my_getter(x):
            return File(path=r[KEY_FILES].get(x.split(':')[-1]), db=db)

        if r.get(KEY_FILES):
            getters.add_getter('file', my_getter)

        if db is not None:
            getters.add_getter('component', lambda x: _get_component(db, x))
            getters.add_getter('blob', _get_blob_callback(db))
            getters.add_getter('file', _get_file_callback(db))

        if schema is not None:
            schema.init()
            r = schema.decode_data(r, getters)

        r = _deep_flat_decode(
            {k: v for k, v in r.items() if k not in (KEY_BUILDS, KEY_BLOBS, KEY_FILES)},
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
        from superduper.base.variables import _find_variables

        return sorted(list(set(_find_variables(self))))

    def set_variables(self, **kwargs) -> 'Document':
        """Set free variables of self.

        :param db: The datalayer to use.
        :param kwargs: The vales to set the variables to `_replace_variables`.
        """
        from superduper.base.variables import _replace_variables

        content = _replace_variables(self, **kwargs)
        return Document(**content)

    def __repr__(self) -> str:
        return f'Document({repr(dict(self))})'

    @staticmethod
    def decode_blobs(schema, r):
        for k, v in schema.fields.items():
            if k not in r:
                continue
            if not isinstance(v, BaseDataType):
                continue
            if v.encodable == 'artifact':
                r[k] = v.decode_data(r[k])
        return r

    def unpack(self, leaves_to_keep: t.Sequence = ()) -> t.Any:
        """Returns the content, but with any encodables replaced by their contents.

        :param leaves_to_keep: The types of leaves to keep.
        """
        out = _unpack(self, leaves_to_keep=leaves_to_keep)
        if self.schema is not None:
            out = self.decode_blobs(self.schema, out)
        if '_base' in out:
            out = out['_base']
        return out

    def __deepcopy__(self, momo):
        new_doc = Document(**self)
        momo[id(self)] = new_doc
        return new_doc


# TODO what is this? Looks like it should be in superduper_mongodb
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

        for mk in (KEY_BUILDS, KEY_FILES, KEY_BLOBS):
            m = original.pop(mk, {})
            for k, v in m.items():
                update[f'{mk}.{k}'] = v

        update = {'$set': update}
        return update

    # TODO needed?
    def to_template(self, **substitutions):
        """
        Convert the document to a template with variables.

        :param substitutions: The substitutions to make.
            `str-to-replace -> variable-name`
        """
        substitutions.setdefault(CFG.output_prefix, 'output_prefix')

        def substitute(x):
            if isinstance(x, str):
                for k, v in substitutions.items():
                    x = x.replace(k, f'<var:{v}>')
                return x
            if isinstance(x, dict):
                return {substitute(k): substitute(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [substitute(v) for v in x]
            return x

        return SuperDuperFlatEncode(substitute(dict(self)))

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
    if isinstance(item, Leaf) and not isinstance(item, tuple(leaves_to_keep)):
        return item.unpack()
    elif isinstance(item, dict):
        return {k: _unpack(v, leaves_to_keep=leaves_to_keep) for k, v in item.items()}
    elif isinstance(item, list):
        return [_unpack(x, leaves_to_keep=leaves_to_keep) for x in item]
    elif isinstance(item, tuple):
        return tuple([_unpack(x, leaves_to_keep=leaves_to_keep) for x in item])
    else:
        return item


def _deep_flat_encode(
    r,
    builds,
    blobs,
    files,
    leaves_to_keep=(),
    metadata: bool = True,
    defaults: bool = True,
    uuids_to_keys: t.Dict | None = None,
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
                metadata=metadata,
                defaults=defaults,
                uuids_to_keys=uuids_to_keys,
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
                    metadata=metadata,
                    defaults=defaults,
                    uuids_to_keys=uuids_to_keys,
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
            metadata=metadata,
            defaults=defaults,
            uuids_to_keys=uuids_to_keys,
        )

    if isinstance(r, Blob):
        blobs[r.identifier] = r.bytes
        return '&:blob:' + r.identifier

    if isinstance(r, File):
        files[r.identifier] = r.path
        return '&:file:' + r.identifier

    # TODO what is this??
    from superduper.backends.base.query import _BaseQuery

    if (
        not isinstance(r, _BaseQuery)
        and getattr(r, 'importable', False)
        and inspect.isfunction(r)
    ):
        ref = r.__name__
        r = dict(r.dict(metadata=metadata, defaults=defaults))
        _deep_flat_encode(
            r,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
            metadata=metadata,
            defaults=defaults,
            uuids_to_keys=uuids_to_keys,
        )
        builds[ref] = r
        return f'?{ref}'

    if isinstance(r, Leaf):
        # If the leaf is component, we need to store the type_id
        type_id = r.type_id if isinstance(r, Component) else None
        key = f"{type_id}:{r.identifier}" if type_id else r.identifier
        uuid = r.uuid

        if key in builds:
            logging.warn(f'Leaf {key} already exists')

        logging.debug(f'Decoding leaf {type(r)} with identifier: {r.identifier}')

        # inline components do not need to be kept
        # they are simply parametrized by their inputs
        if isinstance(r, leaves_to_keep) and not r.inline:
            builds[key] = r
            return '?' + key

        r = r.dict(metadata=metadata, defaults=defaults)
        r = _deep_flat_encode(
            r,
            builds=builds,
            blobs=blobs,
            files=files,
            leaves_to_keep=leaves_to_keep,
            metadata=metadata,
            defaults=defaults,
            uuids_to_keys=uuids_to_keys,
        )

        identifier = r.pop('identifier')
        # Rebuild the key with the identifier
        key = f"{type_id}:{identifier}" if type_id else identifier
        builds[key] = r
        if isinstance(uuids_to_keys, dict):
            uuids_to_keys[key] = uuid
        return f'?{key}'

    return r


def _get_leaf_from_cache(k, builds, getters, db: t.Optional['Datalayer'] = None):
    if reference := parse_reference(f'?{k}'):
        if reference.name in getters:
            out = getters[reference.name](reference.path)
            if reference.attribute is not None:
                return getattr(out, reference.attribute)

    attribute = None
    if '.' in k:
        k, attribute = k.split('.')

    if isinstance(builds[k], Leaf):
        leaf = builds[k]
        if not leaf.db:
            leaf.db = db
        if attribute is not None:
            return getattr(leaf, attribute)
        return leaf

    leaf = _deep_flat_decode(builds[k], builds, getters=getters, db=db)

    builds[k] = leaf
    to_del = []
    keys = list(builds.keys())

    for other in keys:
        import re

        matches = re.findall(f'.*\?\(({k}\..*?)\)', other)
        old_other = other[:]
        if matches:
            for match in matches:
                got = _get_leaf_from_cache(match, builds, getters, db=db)
                other = other.replace(f'?({match})', got)
            builds[other] = builds[old_other]
            to_del.append(old_other)

    for other in to_del:
        del builds[other]

    if isinstance(leaf, Leaf):
        if not leaf.db:
            leaf.db = db

    if attribute is not None:
        return getattr(leaf, attribute)

    return leaf


def _deep_flat_decode(r, builds, getters: Getters, db: t.Optional['Datalayer'] = None):
    if isinstance(r, Leaf):
        return r
    if isinstance(r, (list, tuple)):
        return type(r)(
            [_deep_flat_decode(x, builds, getters=getters, db=db) for x in r]
        )
    if isinstance(r, dict) and '_path' in r and r['_path'] is not None:
        parts = r['_path'].split('.')
        cls = parts[-1]
        module = '.'.join(parts[:-1])
        dict_ = {k: v for k, v in r.items() if k != '_path'}
        dict_ = _deep_flat_decode(dict_, builds, getters=getters, db=db)
        instance = import_item(cls=cls, module=module, dict=dict_, db=db)
        return instance
    if isinstance(r, dict) and '_object' in r:
        dict_ = {k: v for k, v in r.items() if k != '_object'}
        dict_ = _deep_flat_decode(dict_, builds, getters=getters, db=db)
        from superduper.components.datatype import DEFAULT_SERIALIZER

        bytes_ = Blob(identifier=r['_object'].split(':')[-1], db=db).unpack()
        object = DEFAULT_SERIALIZER.decode_data(bytes_)
        instance = import_item(object=object, dict=dict_, db=db)
        return instance
    if isinstance(r, dict):
        literals = r.get('_literals', [])
        return {
            _deep_flat_decode(k, builds, getters=getters, db=db): (
                _deep_flat_decode(v, builds, getters=getters, db=db)
                if k not in literals
                else v
            )
            for k, v in r.items()
        }
    if isinstance(r, str) and '?(' in r:
        import re

        matches = re.findall(r'\?\((.*?)\)', r)
        for match in matches:
            try:
                r = r.replace(
                    f'?({match})', _get_leaf_from_cache(match, builds, getters, db=db)
                )
            except Exception:
                return r
    if isinstance(r, str) and r.startswith('?'):
        out = _get_leaf_from_cache(r[1:], builds, getters=getters, db=db)
        return out
    if isinstance(r, str) and r.startswith('&'):
        assert getters is not None
        reference = parse_reference(r)
        out = getters.run(reference.name, reference.path)
        return out
    return r


def _check_if_version(x):
    if x.isnumeric():
        if int(x) < _VERSION_LIMIT:
            return True
    return False


def _get_component(db, path):
    parts = path.split(':')
    if len(parts) == 1:
        return db.load(uuid=parts[0])
    if len(parts) == 2:
        return db.load(type_id=parts[0], identifier=parts[1])
    if len(parts) == 3:
        if not _check_if_version(parts[2]):
            out = db.load(uuid=parts[2])
            return out
        return db.load(type_id=parts[0], identifier=parts[1], version=parts[2])
    raise ValueError(f'Invalid component reference: {path}')


def _get_file_callback(db):
    def callback(ref):
        return File(identifier=ref, db=db)

    return callback


def _get_blob_callback(db):
    def callback(ref):
        return Blob(identifier=ref, db=db)

    return callback
