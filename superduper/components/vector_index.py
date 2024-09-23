import dataclasses as dc
import itertools
import typing as t

import numpy as np
import tqdm
from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.datatype import DataType
from superduper.components.listener import Listener
from superduper.components.model import Mapping, ModelInputType
from superduper.components.cdc import CDC
from superduper.ext.utils import str_shape
from superduper.jobs.annotations import trigger
from superduper.misc.annotations import component
from superduper.misc.special_dicts import MongoStyleDict
from superduper.vector_search.base import VectorIndexMeasureType, VectorItem

if t.TYPE_CHECKING:
    from superduper.backends.base.cluster import Cluster

KeyType = t.Union[str, t.List, t.Dict]

T = t.TypeVar('T')


def ibatch(iterable: t.Iterable[T], batch_size: int) -> t.Iterator[t.List[T]]:
    """Batch an iterable into chunks of size `batch_size`.

    :param iterable: the iterable to batch
    :param batch_size: the number of groups to write
    """
    iterator = iter(iterable)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def backfill_vector_search(db, vi, searcher):
    """
    Backfill vector search from model outputs of a given vector index.

    :param db: Datalayer instance.
    :param vi: Identifier of vector index.
    :param searcher: FastVectorSearch instance to load model outputs as vectors.
    """
    from superduper.components.datatype import _BaseEncodable

    logging.info(f"Loading vectors of vector-index: '{vi.identifier}'")

    if vi.indexing_listener.select is None:
        raise ValueError('.select must be set')

    outputs_key = vi.indexing_listener.outputs
    query = db[outputs_key].select()

    logging.info(str(query))
    id_field = '_source'

    progress = tqdm.tqdm(desc='Loading vectors into vector-table...')
    notfound = 0
    found = 0
    for record_batch in ibatch(
        db.execute(query),
        CFG.cluster.vector_search.backfill_batch_size,
    ):
        items = []
        for record in record_batch:
            id = record[id_field]
            assert not isinstance(vi.indexing_listener.model, str)
            try:
                h = record[outputs_key]
            except KeyError:
                notfound += 1
                continue
            else:
                found += 1
            if isinstance(h, _BaseEncodable):
                h = h.unpack()
            items.append(VectorItem.create(id=str(id), vector=h))
        if items:
            searcher.add(items)
        progress.update(len(items))

    if notfound:
        logging.warn(
            f'{notfound} document/rows were missing outputs ',
            'key hence skipping vector loading for those.',
        )

    searcher.post_create()
    logging.info(f'Loaded {found} vectors into vector index succesfully')


class VectorIndex(CDC):
    """
    A component carrying the information to apply a vector index.

    :param indexing_listener: Listener which is applied to created vectors
    :param compatible_listener: Listener which is applied to vectors to be compared
    :param measure: Measure to use for comparison
    :param metric_values: Metric values for this index
    """

    type_id: t.ClassVar[str] = 'vector_index'

    indexing_listener: Listener
    compatible_listener: t.Optional[Listener] = None
    measure: VectorIndexMeasureType = VectorIndexMeasureType.cosine
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    cdc_table: str = ''

    def __post_init__(self, db, artifacts):
        return super().__post_init__(db, artifacts)

    def post_create(self, db: Datalayer) -> None:
        self.indexing_listener.model = db.load(uuid=self.indexing_listener.model.uuid)
        super().post_create(db)

    def declare_component(self, cluster: 'Cluster'):
        super().declare_component(cluster)

    def pre_create(self, db: Datalayer) -> None:
        """Called the first time this component is created.

        :param db: the db that creates the component.
        """
        super().pre_create(db)  
        self.cdc_table = self.indexing_listener.outputs

    def __hash__(self):
        return hash((self.type_id, self.identifier))

    def __eq__(self, other: t.Any):
        if isinstance(other, Component):
            return (
                self.identifier == other.identifier and self.type_id and other.type_id
            )
        return False

    # TODO consider a flag such as depends='*' 
    # so that an "apply" trigger runs after all of the other 
    # triggers
    @trigger('apply', 'insert', 'update')
    def copy_vectors(self, ids: t.Sequence[str] | None = None):
        """Copy vectors to the vector index."""
        self.db.cluster.vector_search.put(self)
        select = self.db[self.cdc_table].select()

        # TODO do this using the backfill_vector_search functionality here
        if ids is None:
            assert self.indexing_listener.select is not None
            cur = select.select_ids.execute()
            ids = [r[select.primary_id] for r in cur]
            docs = [r.unpack() for r in select.execute()]
        else:
            docs = [r.unpack() for r in select.select_using_ids(ids).execute()]

        vectors = []
        nokeys = 0
        for doc in docs:
            try:
                vector = MongoStyleDict(doc)[
                    f'{CFG.output_prefix}{self.indexing_listener.predict_id}'
                ]
            except KeyError:
                nokeys += 1
                continue

            vectors.append(
                {
                    'vector': vector,
                    'id': str(doc['_source']),
                }
            )

        if nokeys:
            logging.warn(
                f'{nokeys} outputs were missing. \n'
                'Note: This might happen in case of `VectorIndex` schedule jobs '
                'trigged before model outputs are yet to be computed.'
            )

        for r in vectors:
            if hasattr(r['vector'], 'numpy'):
                r['vector'] = r['vector'].numpy()
        
        # TODO combine logic from backfill
        if vectors:
            searcher = self.db.cluster.vector_search[self.identifier]
            searcher.add(
                [VectorItem(**vector) for vector in vectors]
            )

    @trigger('delete')
    def delete_vectors(self, ids: t.Sequence[str] | None = None):
        """Delete vectors from the vector index."""
        self.db.cluster.vector_search[self.uuid].delete(ids)

    def get_vector(
        self,
        like: Document,
        models: t.List[str],
        keys: KeyType,
        db: t.Any = None,
        outputs: t.Optional[t.Dict] = None,
    ):
        """Peform vector search.

        Perform vector search with query `like` from outputs in db
        on `self.identifier` vector index.

        :param like: The document to compare against
        :param models: List of models to retrieve outputs
        :param keys: Keys available to retrieve outputs of model
        :param db: A datalayer instance.
        :param outputs: (optional) update `like` with outputs

        """
        document = MongoStyleDict(like.unpack())
        if outputs is not None:
            document.update(outputs)
            assert not isinstance(self.indexing_listener, str)
        available_keys = list(document.keys())

        key: t.Optional[t.Any] = None
        model_name: t.Optional[str] = None
        for m, k in zip(models, keys):
            if isinstance(k, str):
                if k in available_keys:
                    model_name, key = m, k
            elif isinstance(k, (tuple, list)):
                if all([i in available_keys for i in list(k)]):
                    model_name, key = m, k
            elif isinstance(k, dict):
                if all([i in available_keys for i in k.values()]):
                    model_name, key = m, k

        if not key:
            try:
                assert isinstance(keys, list)
                kix = keys.index('_base')
                model_name, key = models[kix], keys[kix]
            except ValueError:
                raise Exception(
                    f'Keys in provided {like} don\'t match'
                    f'VectorIndex keys: {keys}, with model: {models}'
                )

        model = db.load('model', model_name)
        data = Mapping(key, model.signature)(document)
        args, kwargs = model.handle_input_type(data, model.signature)
        return (
            model.predict(*args, **kwargs),
            model.identifier,
            key,
        )

    def get_nearest(
        self,
        like: Document,
        db: t.Any,
        id_field: str = '_id',
        outputs: t.Optional[t.Dict] = None,
        ids: t.Optional[t.Sequence[str]] = None,
        n: int = 100,
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Get nearest results in this vector index.

        Given a document, find the nearest results in this vector index, returned as
        two parallel lists of result IDs and scores.

        :param like: The document to compare against
        :param db: The datalayer to use
        :param id_field: Identifier field
        :param outputs: An optional dictionary
        :param ids: A list of ids to match
        :param n: Number of items to return
        """
        models, keys = self.models_keys
        if len(models) != len(keys):
            raise ValueError(f'len(model={models}) != len(keys={keys})')
        within_ids = ids or ()

        searcher = db.cluster.vector_search[self.identifier]

        if isinstance(like, dict) and id_field in like:
            return searcher.find_nearest_from_id(
                str(like[id_field]), within_ids=within_ids, limit=n
            )

        h = self.get_vector(
            like=like,
            models=models,
            keys=keys,
            db=db,
            outputs=outputs,
        )[0]

        return searcher.find_nearest_from_array(h, within_ids=within_ids, n=n)

    def cleanup(self, db: Datalayer):
        """Clean up the vector index.

        :param db: The datalayer to cleanup
        """
        db.cluster.vector_search.drop(self.identifier)

    @property
    def models_keys(self) -> t.Tuple[t.List[str], t.List[ModelInputType]]:
        """Return a list of model and keys for each listener."""
        assert not isinstance(self.indexing_listener, str)
        assert not isinstance(self.compatible_listener, str)

        if self.compatible_listener:
            listeners = [self.indexing_listener, self.compatible_listener]
        else:
            listeners = [self.indexing_listener]

        models = [w.model.identifier for w in listeners]
        keys = [w.key for w in listeners]
        return models, keys

    @property
    def dimensions(self) -> int:
        """Get dimension for vector database.

        This dimension will be used to prepare vectors in the vector database.
        """
        assert not isinstance(self.indexing_listener, str)
        assert not isinstance(self.indexing_listener.model, str)
        if shape := getattr(self.indexing_listener.model.datatype, 'shape', None):
            return shape[-1]
        raise ValueError('Couldn\'t get shape of model outputs from model encoder')

    # def triggerz_ids(self, query: Query, primary_ids: t.Sequence):
    #     """Get trigger IDs.

    #     Only the ids returned by this function will trigger the vector_index.

    #     :param query: Query object.
    #     :param primary_ids: Primary IDs.
    #     """

    #             conditions = [
    #         # trigger by main table
    #         self.select and self.select.table == query.table,
    #         # trigger by output table
    #         query.table in self.key and query.table != self.outputs,
    #     ]
    #     if not isinstance(self.select, Query):
    #         return []

    #     if self.indexing_listener.outputs != query.table:
    #         return []

    #     ids = self.db.databackend.check_ready_ids(
    #         self.select, [self.indexing_listener.outputs], primary_ids
    #     )
    #     return ids


# TODO what is this?
class EncodeArray:
    """Class to encode an array.

    :param dtype: Datatype of array
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x, info: t.Optional[t.Dict] = None):
        """Encode an array.

        :param x: The array to encode
        :param info: Optional info
        """
        x = np.asarray(x)
        if x.dtype != self.dtype:
            raise TypeError(f'dtype was {x.dtype}, expected {self.dtype}')
        return memoryview(x).tobytes()


class DecodeArray:
    """Class to decode an array.

    :param dtype: Datatype of array
    """

    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, bytes, info: t.Optional[t.Dict] = None):
        """Decode an array.

        :param bytes: The bytes to decode
        :param info: Optional info
        """
        return np.frombuffer(bytes, dtype=self.dtype).tolist()


@component(
    {'name': 'shape', 'type': 'int'},
    {'name': 'identifier', 'type': 'str'},
)
def vector(shape, identifier: t.Optional[str] = None):
    """Create an encoder for a vector (list of ints/ floats) of a given shape.

    :param shape: The shape of the vector
    :param identifier: The identifier of the vector
    """
    if isinstance(shape, int):
        shape = (shape,)

    identifier = identifier or f'vector[{str_shape(shape)}]'
    return DataType(
        identifier=identifier,
        shape=shape,
        encoder=None,
        decoder=None,
        encodable='native',
    )


@component()
def sqlvector(shape, bytes_encoding: t.Optional[str] = None):
    """Create an encoder for a vector (list of ints/ floats) of a given shape.

    This is used for compatibility with SQL databases, as the default vector

    :param shape: The shape of the vector
    :param bytes_encoding: The encoding of the bytes
    """
    return DataType(
        identifier=f'sqlvector[{str_shape(shape)}]',
        shape=shape,
        encoder=EncodeArray(dtype='float64'),
        decoder=DecodeArray(dtype='float64'),
        bytes_encoding=bytes_encoding,
    )
