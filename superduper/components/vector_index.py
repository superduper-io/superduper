import dataclasses as dc
import itertools
import typing as t

import numpy as np
import tqdm

from superduper import CFG, logging
from superduper.backends.base.vector_search import VectorIndexMeasureType, VectorItem
from superduper.base.annotations import trigger
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.components.cdc import CDC
from superduper.components.component import Component
from superduper.components.listener import Listener
from superduper.components.model import Mapping, ModelInputType
from superduper.components.schema import Schema
from superduper.components.table import Table
from superduper.misc.special_dicts import MongoStyleDict

if t.TYPE_CHECKING:
    pass

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
    breaks: t.ClassVar[t.Sequence[str]] = ('indexing_listener',)

    indexing_listener: Listener
    compatible_listener: t.Optional[Listener] = None
    measure: VectorIndexMeasureType = VectorIndexMeasureType.cosine
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    cdc_table: str = ''

    def __post_init__(self, db):
        self.cdc_table = self.cdc_table or self.indexing_listener.outputs
        return super().__post_init__(db)

    def refresh(self):
        if self.cdc_table.startswith(CFG.output_prefix):
            self.cdc_table = self.indexing_listener.outputs

    # TODO why this?
    def __hash__(self):
        return hash((self.type_id, self.identifier))

    def __eq__(self, other: t.Any):
        if isinstance(other, Component):
            return (
                self.identifier == other.identifier and self.type_id and other.type_id
            )
        return False

    def _pre_create(self, db: Datalayer, startup_cache: t.Dict = {}):
        assert isinstance(self.indexing_listener, Listener)
        assert hasattr(self.indexing_listener, 'output_table')
        assert hasattr(self.indexing_listener.output_table, 'schema')
        assert isinstance(self.indexing_listener, Listener)
        assert isinstance(self.indexing_listener.output_table, Table)
        try:
            assert isinstance(
                self.indexing_listener.output_table.schema,
                Schema,
            )
            next(
                v
                for v in self.indexing_listener.output_table.schema.fields.values()
                if hasattr(v, 'shape') and v.shape is not None
            )
        except StopIteration:
            raise Exception(
                f'Couldn\'t get a vector shape for '
                f'{self.indexing_listener.output_table.schema.huuid}'
            )

    # TODO consider a flag such as depends='*'
    # so that an "apply" trigger runs after all of the other
    # triggers
    @trigger('apply', 'insert', 'update')
    def copy_vectors(self, ids: t.Sequence[str] | None = None):
        """Copy vectors to the vector index."""
        if not hasattr(self.indexing_listener.model, 'datatype'):
            self.indexing_listener.model = self.db.load(
                uuid=self.indexing_listener.model.uuid
            )
        assert isinstance(self.db, Datalayer)
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
            searcher.add([VectorItem(**vector) for vector in vectors])

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
        super().cleanup(db=db)
        db.cluster.vector_search.drop(self)

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
        msg = f'Couldn\'t find an output table for {self.indexing_listener.huuid}'
        assert isinstance(self.indexing_listener.output_table, Table), msg
        msg = (
            f'Couldn\'t find an output table schema for '
            f'{self.indexing_listener.output_table.huuid}'
        )
        assert hasattr(self.indexing_listener.output_table, 'schema')
        msg = (
            f'Couldn\'t get a vector shape for '
            f'{self.indexing_listener.output_table.schema.huuid}'
        )

        dt = next(
            v
            for v in self.indexing_listener.output_table.schema.fields.values()
            if hasattr(v, 'shape') and v.shape is not None
        )

        try:
            assert dt.shape is not None, msg
            assert isinstance(dt.shape, (tuple, list))
            return dt.shape[-1]
        except IndexError as e:
            raise Exception(
                f'Couldn\'t get a vector shape for {dt.huuid} due to empty shape'
            ) from e


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
