import copy
import dataclasses as dc
import typing as t


from superduperdb import CFG
from superduperdb.backends.base.query import (
    CompoundSelect,
    Delete,
    Insert,
    Like,
    QueryComponent,
    QueryLinker,
    QueryType,
    Select,
    TableOrCollection,
    Update,
)
from superduperdb.base.cursor import SuperDuperCursor
from superduperdb.base.document import Document
from superduperdb.base.serializable import Variable
from superduperdb.misc.files import load_uris


class FindOne(QueryComponent):
    """
    Wrapper around ``astraDB.Collection.find_one``

    :param args: Positional arguments to ``astraDB.Collection.find_one``
    :param kwargs: Named arguments to ``astraDB.Collection.find_one``
    """

    def select_using_ids(self, ids):
        ids = [id for id in ids]
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': {'$in': ids}})
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def add_fold(self, fold: str):
        """
        Modify the query to add a fold to filter {'_fold': fold}

        :param fold: The fold to add
        """

        args = self.args or [{}]
        args[0]['_fold'] = fold
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )


@dc.dataclass
class Find(QueryComponent):
    """
    Wrapper around ``astraDB.Collection.find``

    :param args: Positional arguments to ``astraDB.Collection.find``
    :param kwargs: Named arguments to ``astraDB.Collection.find``
    """

    output_fields: t.Optional[t.Dict[str, str]] = None

    def __post_init__(self):
        if not self.args:
            self.args = [{}]
        else:
            self.args = list(self.args)
        if not self.args[1:]:
            self.args.append({})

        if self.output_fields is not None:
            # if outputs are specified, then project to those outputs
            self.args[1].update(
                {f'_outputs.{k}.{v}': 1 for k, v in self.output_fields.items()}
            )
            if '_id' not in self.args[1]:
                self.args[1]['_id'] = 1

    @property
    def select_ids(self):
        args = list(self.args)[:]
        if not args:
            args = [{}]
        else:
            args = list(self.args)
        if not args[1:]:
            args.append({})

        args[1].update({'_id': 1})
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def outputs(self, **kwargs):
        """
        Join the query with the outputs for a table.

        :param **kwargs: key=model/version or key=model pairs
        """
        args = copy.deepcopy(list(self.args[:]))
        if not args:
            args = [{}]
        if not args[1:]:
            args.append({})

        for k, v in kwargs.items():
            if '/' in v:
                model, version = v.split('/')
                args[1][f'_outputs.{k}.{model}.{version}'] = 1
            else:
                args[1][
                    Variable(
                        f'_outputs.{k}.{v}' + '.{version}',
                        lambda db, value, kwargs: value.format(
                            version=db.show('model', model)[-1]
                        ),
                    )
                ] = 1
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
            output_fields=kwargs,
        )

    def select_using_ids(self, ids):
        ids = [id for id in ids]
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': {'$in': ids}})
        return Find(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

    def select_ids_of_missing_outputs(self, key: str, model: str, version: int):
        assert self.type == QueryType.QUERY
        if self.args:
            args = [
                {
                    '$and': [
                        self.args[0],
                        {f'_outputs.{key}.{model}.{version}': {'$exists': False}},
                    ]
                },
                *self.args[1:],
            ]
        else:
            args = [{f'_outputs.{key}.{model}': {'$exists': False}}]

        if len(args) == 1:
            args.append({})

        args[1] = {'_id': 1}

        return Find(
            name='find',
            type=QueryType.QUERY,
            args=args,
            kwargs=self.kwargs,
        )

    def select_single_id(self, id):
        assert self.type == QueryType.QUERY
        args = list(self.args)[:]
        if not args:
            args = [{}]
        args[0].update({'_id': id})
        return QueryComponent(
            name='find_one',
            type=QueryType.QUERY,
            args=args,
            kwargs=self.kwargs,
        )

    def add_fold(self, fold: str):
        args = self.args
        if not self.args:
            args = [{}]
        args[0]['_fold'] = fold
        return FindOne(
            name=self.name,
            type=self.type,
            args=args,
            kwargs=self.kwargs,
        )

@dc.dataclass(repr=False)
class AstraCompoundSelect(CompoundSelect):
    def _get_query_linker(self, table_or_collection, members) -> 'QueryLinker':
        return AstraQueryLinker(
            table_or_collection=table_or_collection, members=members
        )

    @property
    def output_fields(self):
        return self.query_linker.output_fields

    def outputs(self, **kwargs):
        """
        This method returns a query which joins a query with the outputs
        for a table.

        :param model: The model identifier for which to get the outputs

        >>> q = Collection(...).find(...).outputs('key', 'model_name')

        """
        assert self.query_linker is not None
        return AstraCompoundSelect(
            table_or_collection=self.table_or_collection,
            pre_like=self.pre_like,
            query_linker=self.query_linker.outputs(**kwargs),
            post_like=self.post_like,
        )

    def _execute(self, db):
        similar_scores = None
        query_linker = self.query_linker
        if self.pre_like:
            similar_ids, similar_scores = self.pre_like.execute(db)
            similar_scores = dict(zip(similar_ids, similar_scores))
            if not self.query_linker:
                return similar_ids, similar_scores
            query_linker = query_linker.select_using_ids(similar_ids)

        if not self.post_like:
            return query_linker.execute(db), similar_scores

        assert self.pre_like is None
        cursor = query_linker.select_ids.execute(db)
        query_ids = [str(document[self.primary_id]) for document in cursor]
        similar_ids, similar_scores = self.post_like.execute(db, ids=query_ids)
        similar_scores = dict(zip(similar_ids, similar_scores))

        post_query_linker = self.query_linker.select_using_ids(similar_ids)
        return post_query_linker.execute(db), similar_scores

    def execute(self, db, reference=False):
        output, scores = self._execute(db)
        if isinstance(output, dict):
            if reference and CFG.hybrid_storage:
                load_uris(output, datatypes=db.datatypes)
            return Document.decode(output, db)
        return output

    def download_update(self, db, id: str, key: str, bytes: bytearray) -> None:
        """
        Update to set the content of ``key`` in the document ``id``.

        :param db: The db to query
        :param id: The id to filter on
        :param key:
        :param bytes: The bytes to update
        """
        if self.collection is None:
            raise ValueError('collection cannot be None')
        update = {'$set': {f'{key}._content.bytes': bytes}}
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        return collection.update_one({'_id': id}, update).get('status')

    def check_exists(self, db):
        ...

    @property
    def select_table(self):
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class AstraQueryLinker(QueryLinker):
    @property
    def query_components(self):
        return self.table_or_collection.query_components

    @property
    def output_fields(self):
        out = {}
        for member in self.members:
            if hasattr(member, 'output_fields'):
                out.update(member.output_fields)
        return out

    def add_fold(self, fold):
        new_members = []
        for member in self.members:
            if hasattr(member, 'add_fold'):
                new_members.append(member.add_fold(fold))
            else:
                new_members.append(member)
        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def outputs(self, **kwargs):
        new_members = []
        for member in self.members:
            if hasattr(member, 'outputs'):
                new_members.append(member.outputs(**kwargs))
            else:
                new_members.append(member)

        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    @property
    def select_ids(self):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_ids'):
                new_members.append(member.select_ids)
            else:
                new_members.append(member)

        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_using_ids(self, ids):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_using_ids'):
                new_members.append(member.select_using_ids(ids))
            else:
                new_members.append(member)

        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def _select_ids_of_missing_outputs(self, key: str, model: str, version: int):
        new_members = []
        for member in self.members:
            if hasattr(member, 'select_ids_of_missing_outputs'):
                new_members.append(
                    member.select_ids_of_missing_outputs(key, model, version=version)
                )
        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=new_members,
        )

    def select_single_id(self, id):
        assert (
            len(self.members) == 1
            and self.members[0].type == QueryType.QUERY
            and hasattr(self.members[0].name, 'select_single_id')
        )
        return AstraQueryLinker(
            table_or_collection=self.table_or_collection,
            members=[self.members[0].select_single_id(id)],
        )

    def execute(self, db):
        parent = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        for member in self.members:
            parent = member.execute(parent)
        return parent


@dc.dataclass(repr=False)
class AstraInsert(Insert):
    one: bool = False

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        documents = [r.encode() for r in self.documents]
        response = collection.chunked_insert_many(documents=documents, chunk_size=20)
        inserted_ids = []
        for status in response:
            inserted_ids.extend(status['status']['insertedIds'])
        return inserted_ids

    @property
    def select_table(self):
        return self.table_or_collection.find()


@dc.dataclass(repr=False)
class AstraDelete(Delete):
    one: bool = False

    @property
    def collection(self):
        return self.table_or_collection

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )
        if self.one:
            ids = []
            if '_id' in self.kwargs:
                ids = [str(self.kwargs['_id'])]
            for arg in self.args:
                if isinstance(arg, dict) and '_id' in arg:
                    ids = [str(arg['_id'])]
            result = None
            if ids:
                result = collection.delete_one(id=ids[0])
            if result.get('status')['deletedCount'] == 1:
                if not ids:
                    deleted_document = collection.find_one(*self.args, **self.kwargs)
                    return [str(deleted_document.get('data')['document']['_id'])]
                return ids
            else:
                return []

        deleted_ids_results = []
        response_generator = collection.paginated_find(
            *self.args, **self.kwargs
        )
        for document in response_generator:
            deleted_ids_results.append(document['_id'])

        more_data = True
        while more_data:
            # Call the delete_many method to delete records based on the filter
            response = collection.delete_many(*self.args, **self.kwargs) 
            # Check if there is more data to delete
            more_data = response.get('status', {}).get('moreData', False)
        return deleted_ids_results


@dc.dataclass(repr=False)
class AstraUpdate(Update):
    update: Document
    filter: t.Dict
    one: bool = False
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def select_table(self):
        return self.table_or_collection.find()

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )

        update = self.update
        if isinstance(self.update, Document):
            update = update.encode()

        if self.one:
            result = collection.find_one(self.filter, {'_id': 1})
            id = result.get('data')['document']['_id']
            collection.update_one({'_id': id}, update)
            return [id]

        response_generator = collection.paginated_find(self.filter, {'_id': 1})
        ids = [document['_id'] for document in response_generator]
        collection.update_many({'_id': {'$in': ids}}, update)
        return ids


@dc.dataclass(repr=False)
class AstraReplaceOne(Update):
    replacement: Document
    filter: t.Dict
    args: t.Sequence = dc.field(default_factory=list)
    kwargs: t.Dict = dc.field(default_factory=dict)

    @property
    def collection(self):
        return self.table_or_collection

    @property
    def select_table(self):
        return self.table_or_collection.find()

    def execute(self, db):
        collection = db.databackend.get_table_or_collection(
            self.table_or_collection.identifier
        )

        replacement = self.replacement
        if isinstance(replacement, Document):
            replacement = replacement.encode()

        result = collection.find_one(filter=self.filter, projection={'_id': 1})
        id = result.get('data')['document']['_id']
        collection.find_one_and_replace(filter={'_id': id},replacement=replacement)
        return [id]

@dc.dataclass(repr=False)
class Collection(TableOrCollection):
    query_components: t.ClassVar[t.Dict] = {'find': Find, 'find_one': FindOne}
    type_id: t.ClassVar[str] = 'collection'

    primary_id: t.ClassVar[str] = '_id'

    def get_table(self, db):
        collection = db.databackend.get_table_or_collection(self.collection.identifier)
        return collection
    
    def _get_query_linker(self, members) -> AstraQueryLinker:
        return AstraQueryLinker(members=members, table_or_collection=self)

    def _get_query(
        self,
        pre_like: t.Optional[Like] = None,
        query_linker: t.Optional[QueryLinker] = None,
        post_like: t.Optional[Like] = None,
        i: int = 0,
    ) -> AstraCompoundSelect:
        return AstraCompoundSelect(
            pre_like=pre_like,
            query_linker=query_linker,
            post_like=post_like,
            table_or_collection=self,
        )

    def _delete(self, *args, one: bool = False, **kwargs):
        return AstraDelete(args=args, kwargs=kwargs, table_or_collection=self, one=one)

    def _insert(self, documents, **kwargs):
        return AstraInsert(documents=documents, kwargs=kwargs, table_or_collection=self)

    def _update(self, filter, update, *args, one: bool = False, **kwargs):
        return AstraUpdate(
            filter=filter,
            update=update,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
            one=one,
        )

    def delete_one(self, *args, **kwargs):
        return self._delete(*args, one=True, **kwargs)

    def replace_one(self, filter, replacement, *args, **kwargs):
        return AstraReplaceOne(
            filter=filter,
            replacement=replacement,
            args=args,
            kwargs=kwargs,
            table_or_collection=self,
        )

    def update_one(self, filter, update, *args, **kwargs):
        return self._update(filter, update, *args, one=True, **kwargs)

    def update_many(self, filter, update, *args, **kwargs):
        return self._update(filter, update, *args, one=False, **kwargs)

    def insert(self, *args, **kwargs):
        return self.insert_many(*args, **kwargs)

    def insert_many(self, *args, **kwargs):
        return self._insert(*args, **kwargs)

    def insert_one(self, document, *args, **kwargs):
        return self._insert([document], *args, **kwargs)

    def model_update(
        self,
        db,
        ids: t.List[t.Any],
        key: str,
        model: str,
        version: int,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        document_embedded = kwargs.get('document_embedded', True)

        if key.startswith('_outputs'):
            key = key.split('.')[1]
        if not outputs:
            return
        if document_embedded:
            if flatten:
                raise AttributeError(
                    'Flattened outputs cannot be stored along with input documents.'
                    'Please use `document_embedded = False` option with flatten = True'
                )
            assert self.collection is not None
            collection = db.databackend.get_table_or_collection(self.identifier)
            for i, id in enumerate(ids):
                collection.update_one(filter={'_id': id},
                                      update={'$set': {f'_outputs.{key}.{model}.{version}': outputs[i]}})
        else:
            if flatten:
                bulk_docs = []
                for i, id in enumerate(ids):
                    _outputs = outputs[i]
                    if isinstance(_outputs, (list, tuple)):
                        for offset, output in enumerate(_outputs):
                            bulk_docs.append(
                                    {
                                        '_outputs': {
                                            key: {model: {str(version): output}}
                                        },
                                        '_source': id,
                                        '_offset': offset,
                                    }
                            )
                    else:
                        bulk_docs.append(
                                {
                                    '_outputs': {
                                        key: {model: {str(version): _outputs}}
                                    },
                                    '_source': id,
                                    '_offset': 0,
                                }
                        )

            else:
                bulk_docs = [
                        {
                            '_id': id,
                            '_outputs': {key: {model: {str(version): outputs[i]}}},
                        }
                    for i, id in enumerate(ids)
                ]

            collection_name = f'_outputs.{key}.{model}'
            collection = db.databackend.get_table_or_collection(collection_name)
            response = collection.chunked_insert_many(documents=bulk_docs, chunk_size=20)
            inserted_ids = []
            for res in response:
                inserted_ids.extend(res['status']['insertedIds'])
