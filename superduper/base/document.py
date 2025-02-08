import inspect
import re
import typing as t
from collections import namedtuple

from superduper import CFG, logging
from superduper.backends.base.artifacts import ArtifactStore
from superduper.base.constant import (
    KEY_BLOBS,
    KEY_BUILDS,
    KEY_FILES,
)
from superduper.base.leaf import Leaf
from superduper.base.variables import _replace_variables
from superduper.components.schema import Schema, get_schema
from superduper.misc.special_dicts import MongoStyleDict, SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


LeafMetaType = t.Type['Leaf']


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


class _InMemoryArtifactStore(ArtifactStore):
    def __init__(self, blobs, files):
        self.blobs = blobs
        self.files = files

    def url(self):
        """Artifact store connection url."""
        raise NotImplementedError

    def _delete_bytes(self, file_id: str):
        """Delete artifact from artifact store.

        :param file_id: File id uses to identify artifact in store
        """
        raise NotImplementedError

    def drop(self, force: bool = False):
        """
        Drop the artifact store.

        :param force: If ``True``, don't ask for confirmation
        """
        raise NotImplementedError

    def _exists(self, file_id: str):
        return file_id in self.files

    def put_bytes(self, serialized: bytes, file_id: str):
        """Save bytes in artifact store.

        :param serialized: Bytes to save
        :param file_id: Identifier of artifact in the store
        """
        raise NotImplementedError

    def put_file(self, file_path: str, file_id: str) -> str:
        """Save file in artifact store and return file_id.

        :param file_path: Path to file
        :param file_id: Identifier of artifact in the store
        """
        raise NotImplementedError

    def delete_artifact(self, artifact_ids: t.List[str]):
        """Delete artifact from artifact store.

        :param artifact_ids: list of artifact ids to delete.
        """
        for artifact_id in artifact_ids:
            try:
                self._delete_bytes(artifact_id)
            except FileNotFoundError:
                logging.warn(f'Blob {artifact_id} not found in artifact store')

    def get_bytes(self, file_id: str) -> bytes:
        """
        Load bytes from artifact store.

        :param file_id: Identifier of artifact in the store
        """
        return self.blobs[file_id]

    def get_file(self, file_id: str) -> str:
        """
        Load file from artifact store and return path.

        :param file_id: Identifier of artifact in the store
        """
        return self.files[file_id]

    def disconnect(self):
        """Disconnect the client."""
        pass


class _TmpDB:
    """Temporary datalayer for decoding documents.

    :param artifact_store: The artifact store to use.
    :param databackend: The databackend to use.
    """

    def __init__(self, artifact_store, databackend):
        self.artifact_store = artifact_store
        self.databackend = databackend

    def __getitem__(self, item):
        from superduper.backends.base.query import Query

        return Query(table=item, parts=(), db=self)


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
        """Map a function over the document.

        :param fn: The function to map.
        :param condition: The condition to map over.
        """

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
        """Update document with values from other.

        :param other: The other document to update with.
        """
        schema = self.schema or Schema('tmp', fields={})

        if isinstance(other, Document) and other.schema:
            assert other.schema is not None
            schema += other.schema
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
        :param metadata: Whether to include metadata.
        :param defaults: Whether to include defaults.
        :param keep_schema: Whether to keep the schema.
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

        out.update({KEY_BUILDS: builds, KEY_FILES: files, KEY_BLOBS: blobs})
        out = SuperDuperFlatEncode(out)
        return out

    def __getitem__(self, key: str) -> t.Any:
        if not key.startswith(CFG.output_prefix) or '.' in key:
            return super().__getitem__(key)

        if re.match(f'{CFG.output_prefix}[^_]{1,}__[a-z0-9]{10,}', key):
            return super().__getitem__(key)

        key = next(k for k in self.keys() if k.startswith(key))
        return super().__getitem__(key)

    @classmethod
    def build_in_memory_db(cls, blobs, files):
        return _TmpDB(
            artifact_store=_InMemoryArtifactStore(blobs=blobs, files=files),
            databackend=namedtuple('tmp_databackend', field_names=('bytes_encoding',))(
                bytes_encoding='bytes'
            ),
        )

    @classmethod
    def decode(
        cls,
        r,
        schema: t.Optional['Schema'] = None,
        db: t.Optional['Datalayer'] = None,
    ):
        """Converts any dictionary into a Document or a Leaf.

        :param r: The encoded data.
        :param schema: The schema to use.
        :param db: The datalayer to use.
        :param getters: The getters to use.
        """
        # TODO - is this the right place for this?

        if db is None:
            blobs = r.pop('_blobs', {})
            files = r.pop('_files', {})
            db = cls.build_in_memory_db(blobs=blobs, files=files)

        if '_variables' in r:
            variables = {**r['_variables'], 'output_prefix': CFG.output_prefix}
            r = _replace_variables(
                {k: v for k, v in r.items() if k != '_variables'}, **variables
            )

        builds = r.get(KEY_BUILDS, {})

        for k in builds:
            if isinstance(builds[k], dict) and (
                '_path' in builds[k] or '_object' in builds[k]
            ):
                builds[k]['identifier'] = k.split(':')[-1]

        # TODO add _path and _object as constants
        if '_path' in r or '_object' in r:
            # TODO this has no place here
            # this should be Component.decode
            # or db.load
            assert '_path' in r or '_object' in r

            if '_path' in r:
                cls = Leaf.get_cls_from_path(r['_path'])
            else:
                assert '_object' in r
                cls = Leaf.get_cls_from_blob(r['_object'], db=db)

            if inspect.isclass(cls):
                schema = cls.build_class_schema(db)
                assert isinstance(schema, Schema)
                r = schema.decode_data(r, builds=builds)

            if inspect.isclass(cls):
                return cls.from_dict(r, db)
            else:
                # TODO the only time this occurs is with BaseQuery
                # Add documents and query as parameters to BaseQuery
                # Then deprecate this option
                r = {
                    k: v for k, v in r.items() if k in inspect.signature(cls).parameters
                }
                return cls(**r, db=db)
        else:
            if schema is None:
                from superduper.base.datalayer import Datalayer

                assert isinstance(db, Datalayer)
                schema: Schema = db.load('schema', r['_schema'])

            assert schema is not None
            schema: Schema = schema.reconnect(db=db)
            r = schema.decode_data(r, builds=builds)

        # TODO don't need these 2 options
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

        :param kwargs: The vales to set the variables to `_replace_variables`.
        """
        from superduper.base.variables import _replace_variables

        content = _replace_variables(self, **kwargs)
        return Document(**content)

    def __repr__(self) -> str:
        return f'Document({repr(dict(self))})'

    def unpack(self, leaves_to_keep: t.Sequence = ()) -> t.Any:
        """Returns the content, but with any encodables replaced by their contents.

        :param leaves_to_keep: The types of leaves to keep.
        """
        return _unpack(self, leaves_to_keep=leaves_to_keep)

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


def _unpack(item: t.Any, leaves_to_keep: t.Sequence = ()) -> t.Any:
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
