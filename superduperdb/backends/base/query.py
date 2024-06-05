import dataclasses as dc
import json
import re
import typing as t
import uuid
from abc import ABC, abstractmethod
from functools import wraps

from superduperdb.base.document import Document
from superduperdb.base.leaf import Leaf
from superduperdb.misc.annotations import merge_docstrings
from superduperdb.misc.hash import hash_string

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer
    from superduperdb.components.schema import Schema


def applies_to(*flavours):
    """Decorator to check if the query matches the accepted flavours.

    :param flavours: The flavours to check against.
    """

    def decorator(f):
        @wraps(f)
        def decorated(self, *args, **kwargs):
            msg = (
                f'Query {self} does not match any of accepted patterns {flavours},'
                f' for the {f.__name__} method to which this method applies.'
            )

            try:
                flavour = self.flavour
            except TypeError:
                raise TypeError(msg)
            assert flavour in flavours, msg
            return f(self, *args, **kwargs)

        return decorated

    return decorator


@merge_docstrings
@dc.dataclass
class _BaseQuery(Leaf, ABC):
    ...


class TraceMixin:
    """Mixin to add trace functionality to a query # noqa."""

    def __getattr__(self, item):
        item = type(self).methods_mapping.get(item, item)
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[*self.parts, item],
        )

    def __call__(self, *args, **kwargs):
        """Add a method call to the query.

        :param args: The arguments to pass to the method.
        :param kwargs: The keyword arguments to pass to the method.
        """
        assert isinstance(self.parts[-1], str)
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=[*self.parts[:-1], (self.parts[-1], args, kwargs)],
        )


@merge_docstrings
@dc.dataclass(kw_only=True, repr=False)
class Query(_BaseQuery, TraceMixin):
    """A query object.

    This base class is used to create a query object that can be executed
    in the datalayer.

    :param parts: The parts of the query.
    """

    flavours: t.ClassVar[t.Dict[str, str]] = {}
    methods_mapping: t.ClassVar[t.Dict[str, str]] = {}

    parts: t.Sequence[t.Union[t.Tuple, str]] = ()

    def __getitem__(self, item):
        if not isinstance(item, slice):
            return super().__getitem__(item)
        assert isinstance(item, slice)
        parts = self.parts[item]
        return type(self)(db=self.db, identifier=self.identifier, parts=parts)

    @property
    def leaves(self):
        return {}

    def set_db(self, db: 'Datalayer'):
        """Set the datalayer to use to execute the query.

        :param db: The datalayer to use to execute the query.
        """

        def _set_db(r, db):
            if isinstance(r, (tuple, list)):
                out = [_set_db(x, db) for x in r]
                return out
            if isinstance(r, Document):
                return Document({k: _set_db(v, db) for k, v in r.items()})
            if isinstance(r, dict):
                return {k: _set_db(v, db) for k, v in r.items()}
            if isinstance(r, Query):
                r.db = db
                return r

            return r

        self.db = db

        # Recursively set db
        parts: t.List[t.Union[str, tuple]] = []
        for part in self.parts:
            if isinstance(part, str):
                parts.append(part)
                continue
            part_args = tuple(_set_db(part[1], db))
            part_kwargs = _set_db(part[2], db)
            part = part[0]
            parts.append((part, part_args, part_kwargs))
        self.parts = parts

    def _get_flavour(self):
        _query_str = self._to_str()
        repr_ = _query_str[0]

        if repr_ == self.identifier and not (_query_str[0] and _query_str[-1]):
            # Table selection query.
            return 'select'

        try:
            return next(k for k, v in self.flavours.items() if re.match(v, repr_))
        except StopIteration:
            raise TypeError(
                f'Query flavour {repr_} did not match existing {type(self)} flavours'
            )

    def _get_parent(self):
        return self.db.databackend.get_table_or_collection(self.identifier)

    def _prepare_pre_like(self, parent):
        like_args, like_kwargs = self.parts[0][1:]
        like_args = list(like_args)
        if not like_args:
            like_args = [{}]
        like = like_args[0] or like_kwargs.pop('r', {})
        if isinstance(like, Document):
            like = like.unpack()

        ids = like_kwargs.pop('within_ids', [])

        n = like_kwargs.pop('n', 100)

        vector_index = like_kwargs.get('vector_index')

        similar_ids, similar_scores = self.db.select_nearest(
            like,
            vector_index=vector_index,
            ids=ids,
            n=n,
        )
        similar_scores = dict(zip(similar_ids, similar_scores))
        return similar_ids, similar_scores

    @property
    def flavour(self):
        """Return the flavour of the query."""
        return self._get_flavour()

    @property
    @abstractmethod
    def documents(self) -> t.List[Document]:
        """Return the documents of the query."""
        pass

    @property
    @abstractmethod
    def type(self):
        """Return the type of the query.

        The type is used to route the correct method to execute the query in the
        datalayer.
        """
        pass

    @property
    def _id(self):
        return f'query/{hash_string(str(self))}'

    def _deep_flat_encode(
        self,
        cache,
        blobs,
        files,
        leaves_to_keep=(),
        schema: t.Optional['Schema'] = None,
    ):
        query, documents = self._dump_query()
        documents = [Document(r) for r in documents]
        for i, doc in enumerate(documents):
            documents[i] = doc._deep_flat_encode(
                cache, blobs, files, leaves_to_keep, schema=schema
            )
        cache[self._id] = {
            '_path': 'superduperdb/backends/base/query/parse_query',
            'documents': documents,
            'query': query,
        }
        return f'?{self._id}'

    @staticmethod
    def _update_item(a, documents, queries):
        if isinstance(a, Query):
            a, sub_documents, sub_queries = a._to_str()
            documents.update(sub_documents)
            queries.update(sub_queries)
            id_ = uuid.uuid4().hex[:5].upper()
            queries[id_] = a
            arg = f'query[{id_}]'
        else:
            id_ = uuid.uuid4().hex[:5].upper()
            if isinstance(a, dict):
                documents[id_] = a
                arg = f'documents[{id_}]'
            elif isinstance(a, list):
                documents[id_] = {'_base': a}
                arg = f'documents[{id_}]'
            else:
                try:
                    arg = json.dumps(a)
                except Exception:
                    documents[id_] = {'_base': a}
                    arg = id_
        return arg

    def _to_str(self):
        documents = {}
        queries = {}
        out = str(self.identifier)
        for part in self.parts:
            if isinstance(part, str):
                out += f'.{part}'
                continue
            args = []
            for a in part[1]:
                args.append(self._update_item(a, documents, queries))
            args = ', '.join(args)
            kwargs = {}
            for k, v in part[2].items():
                kwargs[k] = self._update_item(v, documents, queries)
            kwargs = ', '.join([f'{k}={v}' for k, v in kwargs.items()])
            if part[1] and part[2]:
                out += f'.{part[0]}({args}, {kwargs})'
            if not part[1] and part[2]:
                out += f'.{part[0]}({kwargs})'
            if part[1] and not part[2]:
                out += f'.{part[0]}({args})'
            if not part[1] and not part[2]:
                out += f'.{part[0]}()'
        return out, documents, queries

    def _dump_query(self):
        output, documents, queries = self._to_str()
        if queries:
            output = '\n'.join(list(queries.values())) + '\n' + output
        for i, k in enumerate(queries):
            output = output.replace(k, str(i))
        for i, k in enumerate(documents):
            output = output.replace(k, str(i))
        documents = list(documents.values())
        return output, documents

    def __repr__(self):
        output, docs = self._dump_query()
        # for i, doc in enumerate(docs):
        #     doc_string = str(doc)
        #     if isinstance(doc, Document):
        #         doc_string = str(doc.unpack())
        #     output = output.replace(f'documents[{i}]', doc_string)
        return output

    def __eq__(self, other):
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=self.parts + [('__eq__', (other,), {})],
        )

    def __lt__(self, other):
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=self.parts + [('__lt__', (other,), {})],
        )

    def __gt__(self, other):
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=self.parts + [('__gt__', (other,), {})],
        )

    def __le__(self, other):
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=self.parts + [('__le__', (other,), {})],
        )

    def __ge__(self, other):
        return type(self)(
            db=self.db,
            identifier=self.identifier,
            parts=self.parts + [('__ge__', (other,), {})],
        )

    def _encode_or_unpack_args(self, r, db, method='encode', parent=None):
        if isinstance(r, Document):
            out = getattr(r, method)()
            from superduperdb.misc.special_dicts import SuperDuperFlatEncode

            if isinstance(out, SuperDuperFlatEncode):
                out.pop_leaves()
                out.pop_files()
                out.pop_blobs()
            return out

        if isinstance(r, (tuple, list)):
            out = [
                self._encode_or_unpack_args(x, db, method=method, parent=parent)
                for x in r
            ]
            if isinstance(r, tuple):
                return tuple(out)
            return out
        if isinstance(r, dict):
            return {
                k: self._encode_or_unpack_args(v, db, method=method, parent=parent)
                for k, v in r.items()
            }
        if isinstance(r, Query):
            r.db = db
            parent = r._get_parent()
            return super(type(self), r)._execute(parent, method=method)

        return r

    def _execute(self, parent, method='encode'):
        for part in self.parts:
            if isinstance(part, str):
                parent = getattr(parent, part)
                continue
            args = self._encode_or_unpack_args(
                part[1], self.db, method=method, parent=parent
            )
            kwargs = self._encode_or_unpack_args(
                part[2], self.db, method=method, parent=parent
            )
            parent = getattr(parent, part[0])(*args, **kwargs)
        return parent

    @abstractmethod
    def _create_table_if_not_exists(self):
        pass

    def execute(self, db=None, **kwargs):
        """
        Execute the query.

        :param db: Datalayer instance.
        """
        self.db = db or self.db
        return self.db.execute(self, **kwargs)

    def do_execute(self, db=None):
        """
        Execute the query.

        This methold will first create the table if it does not exist and then
        execute the query.

        All the methods matching the pattern `_execute_{flavour}` will be
        called if they exist.

        If no such method exists, the `_execute` method will be called.

        :param db: The datalayer to use to execute the query.
        """
        self.db = db or self.db
        assert self.db is not None, 'No datalayer (db) provided'
        self._create_table_if_not_exists()
        parent = self._get_parent()
        try:
            flavour = self._get_flavour()
            handler = f'_execute_{flavour}' in dir(self)
            if handler is False:
                raise AssertionError
            handler = getattr(self, f'_execute_{flavour}')
            return handler(parent=parent)
        except TypeError as e:
            if 'did not match' in str(e):
                return self._execute(parent=parent)
            else:
                raise e
        except AssertionError:
            return self._execute(parent=parent)

    @property
    @abstractmethod
    def primary_id(self):
        """Return the primary id of the table."""
        pass

    @abstractmethod
    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        """Update the model outputs in the database.

        :param ids: The ids of the documents to update.
        :param predict_id: The id of the prediction.
        :param outputs: The outputs to store.
        :param flatten: Whether to flatten the outputs.
        :param kwargs: Additional keyword arguments.
        """
        pass

    @abstractmethod
    def add_fold(self, fold: str):
        """Add a fold to the query.

        :param fold: The fold to add.
        """
        pass

    @abstractmethod
    def select_using_ids(self, ids: t.Sequence[str]):
        """Return a query that selects ids.

        :param ids: The ids to select.
        """
        pass

    @property
    @abstractmethod
    def select_ids(self, ids: t.Sequence[str]):
        """Return a query that selects ids.

        :param ids: The ids to select.
        """
        pass

    @abstractmethod
    def select_ids_of_missing_outputs(self, predict_id: str):
        """Return the ids of missing outputs.

        :param predict_id: The id of the prediction.
        """
        pass

    @abstractmethod
    def select_single_id(self, id: str):
        """Return a single document by id.

        :param id: The id of the document.
        """
        pass

    @property
    @abstractmethod
    def select_table(self):
        """Return the table to select from."""
        pass

    def _prepare_documents(self):
        documents = self.documents
        kwargs = self.parts[0][2]
        schema = kwargs.pop('schema', None)

        if schema is None:
            try:
                table = self.db.tables[self.identifier]
                schema = table.schema
            except FileNotFoundError:
                pass

        documents = [
            r.encode(schema) if isinstance(r, Document) else r for r in documents
        ]
        for r in documents:
            r = self.db.artifact_store.save_artifact(r)
        return documents

    @property
    def table_or_collection(self):
        """Return the table or collection to select from."""
        return type(self)(identifier=self.identifier, db=self.db)


def _parse_query_part(part, documents, query, builder_cls, db=None):
    key = part.split('.')
    if key[0] == '_outputs':
        identifier = f'{key[0]}.{key[1]}'
        part = part.split('.')[2:]
    else:
        identifier = key[0]
        part = part.split('.')[1:]

    current = builder_cls(identifier=identifier, parts=(), db=db)
    for comp in part:
        match = re.match('^([a-zA-Z0-9_]+)\((.*)\)$', comp)
        if match is None:
            current = getattr(current, comp)
            continue
        if not match.groups()[1].strip():
            current = getattr(current, match.groups()[0])()
            continue

        comp = getattr(current, match.groups()[0])
        args_kwargs = [x.strip() for x in match.groups()[1].split(',')]
        args = []
        kwargs = {}
        for x in args_kwargs:
            if '=' in x:
                k, v = x.split('=')
                kwargs[k] = eval(v, {'documents': documents, 'query': query})
            else:
                args.append(eval(x, {'documents': documents, 'query': query}))
        current = comp(*args, **kwargs)
    return current


def parse_query(
    query: t.Union[str, list],
    builder_cls: t.Optional[t.Type[Query]] = None,
    documents: t.Sequence[t.Any] = (),
    db: t.Optional['Datalayer'] = None,
):
    """Parse a string query into a query object.

    :param query: The query to parse.
    :param builder_cls: The class to use to build the query.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """
    if isinstance(query, str) and query.split('\n')[-1].strip().split('.')[
        1
    ].startswith('predict'):
        builder_cls = Model
        return _parse_query_part(query, documents, [], builder_cls, db=db)

    documents = [Document(r, db=db) for r in documents]
    if isinstance(query, str):
        query = [x.strip() for x in query.split('\n') if x.strip()]
    for i, q in enumerate(query):
        query[i] = _parse_query_part(q, documents, query[:i], builder_cls, db=db)
    return query[-1]


@merge_docstrings
@dc.dataclass
class Model(Leaf, TraceMixin):
    """
    A model helper class for create a query to predict.

    :param parts: The parts of the query.
    """

    parts: t.Sequence[t.Union[t.Tuple, str]] = ()
    methods_mapping: t.ClassVar[t.Dict[str, str]] = {}
    type: t.ClassVar[str] = 'predict'

    def execute(self):
        """Execute the model as a query."""
        return self.db.execute(self)

    def do_execute(self, db=None):
        """Execute the query.

        :param db: Datalayer instance.
        """
        self.db = db
        m = self.db.models[self.identifier]
        method = getattr(m, self.parts[-1][0])
        r = method(*self.parts[-1][1], **self.parts[-1][2])
        return Document({'_base': r})
