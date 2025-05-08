import dataclasses as dc
import itertools
import typing as t

import tqdm

from superduper import CFG, logging
from superduper.backends.base.vector_search import VectorItem
from superduper.base.annotations import trigger
from superduper.base.datalayer import Datalayer
from superduper.base.document import Document
from superduper.base.schema import Schema
from superduper.components.cdc import CDC
from superduper.components.listener import Listener
from superduper.components.table import Table
from superduper.misc.special_dicts import DeepKeyedDict

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


class VectorIndex(CDC):
    """
    A component carrying the information to apply a vector index.

    :param indexing_listener: Listener which is applied to created vectors
    :param compatible_listener: Listener which is applied to vectors to be compared
    :param measure: Measure to use for comparison
    :param metric_values: Metric values for this index
    :param cdc_table: Table to use for CDC
    """

    breaks: t.ClassVar[t.Sequence[str]] = ('indexing_listener',)

    indexing_listener: Listener
    compatible_listener: t.Optional[Listener] = None
    measure: str = 'cosine'
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)
    cdc_table: str = ''

    def postinit(self):
        """Post-initialization method."""
        self.cdc_table = self.cdc_table or self.indexing_listener.outputs

        assert isinstance(self.indexing_listener, Listener)
        assert hasattr(self.indexing_listener, 'output_table')
        assert hasattr(self.indexing_listener.output_table, 'schema')
        assert isinstance(self.indexing_listener, Listener)
        assert isinstance(self.indexing_listener.output_table, Table)
        try:
            next(
                v
                for v in self.indexing_listener.output_table.schema.fields.values()
                if hasattr(v, 'shape') and v.shape is not None
            )
        except StopIteration:
            raise Exception(
                f'Couldn\'t get a vector shape for\n'
                f'{self.indexing_listener.output_table.schema}'
            )

        super().postinit()

    def on_create(self):
        """Declare the component to the cluster."""
        super().on_create()
        self.db.cluster.vector_search.put_component(self)

    def get_vectors(self, ids: t.Sequence[str] | None = None):
        """Get vectors from the vector index.

        :param ids: A list of ids to match
        """
        if not hasattr(self.indexing_listener.model, 'datatype'):
            self.indexing_listener.model = self.db.load(
                uuid=self.indexing_listener.model.uuid
            )
        assert isinstance(self.db, Datalayer)
        select = self.db[self.cdc_table].select()

        # TODO do this using the backfill_vector_search functionality here
        if ids is None:
            assert self.indexing_listener.select is not None
            ids = select.ids()
            docs = [r.unpack() for r in select.execute()]
        else:
            docs = [r.unpack() for r in select.subset(ids)]

        vectors = []
        nokeys = 0
        for doc in docs:
            try:
                vector = DeepKeyedDict(doc)[
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

        return vectors

    # TODO consider a flag such as depends='*'
    # so that an "apply" trigger runs after all of the other
    # triggers
    @trigger('apply', 'insert', 'update')
    def copy_vectors(self, ids: t.Sequence[str] | None = None):
        """Copy vectors to the vector index."""
        vectors = self.get_vectors(ids=ids)
        # TODO combine logic from backfill
        if vectors:
            self.db.cluster.vector_search.add(
                uuid=self.uuid, vectors=[VectorItem(**vector) for vector in vectors]
            )

    @trigger('delete')
    def delete_vectors(self, ids: t.Sequence[str] | None = None):
        """Delete vectors from the vector index."""
        self.db.cluster.vector_search[self.uuid].delete(ids)

    # TODO refactor to improve readability
    def get_vector(
        self,
        like: Document,
        models: t.Dict,
        keys: KeyType,
        outputs: t.Optional[t.Dict] = None,
    ):
        """Peform vector search.

        Perform vector search with query `like` from outputs in db
        on `self.identifier` vector index.

        :param like: The document to compare against
        :param models: List of models to retrieve outputs
        :param keys: Keys available to retrieve outputs of model
        :param outputs: (optional) update `like` with outputs

        """
        document = DeepKeyedDict(like.unpack())
        if outputs is not None:
            document.update(outputs)
            assert not isinstance(self.indexing_listener, str)
        available_keys = list(document.keys())

        key: t.Optional[t.Any] = None
        model_name: t.Optional[str] = None
        for m, k in zip(list(models.keys()), keys):
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

        model = models[model_name]
        assert model.signature == 'singleton'
        return (
            model.predict(document[key]),
            model.identifier,
            key,
        )

    def get_nearest(
        self,
        like: Document,
        outputs: t.Optional[t.Dict] = None,
        ids: t.Optional[t.Sequence[str]] = None,
        n: int = 100,
    ) -> t.Tuple[t.List[str], t.List[float]]:
        """Get nearest results in this vector index.

        Given a document, find the nearest results in this vector index, returned as
        two parallel lists of result IDs and scores.

        :param like: The document to compare against
        :param outputs: An optional dictionary
        :param ids: A list of ids to match
        :param n: Number of items to return
        """
        models, keys = self.models_keys
        if len(models) != len(keys):
            raise ValueError(f'len(model={models}) != len(keys={keys})')
        within_ids = ids or ()

        h = self.get_vector(
            like=like,
            models=models,
            keys=keys,
            outputs=outputs,
        )[0]

        return self.db.cluster.vector_search.find_nearest_from_array(
            component=self.component,
            vector_index=self.identifier,
            h=h,
            n=n,
            within_ids=within_ids,
        )

    def cleanup(self):
        """Clean up the vector index."""
        super().cleanup()
        self.db.cluster.vector_search.drop_component(self.component, self.identifier)

    @property
    def models_keys(self):
        """Return a list of model and keys for each listener."""
        assert not isinstance(self.indexing_listener, str)
        assert not isinstance(self.compatible_listener, str)

        if self.compatible_listener:
            listeners = [self.indexing_listener, self.compatible_listener]
        else:
            listeners = [self.indexing_listener]

        models = {w.model.identifier: w.model for w in listeners}
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
            f'{self.indexing_listener.output_table.schema}'
        )

        dt = next(
            v
            for v in self.indexing_listener.output_table.schema.fields.values()
            if hasattr(v, 'shape') and v.shape is not None
        )

        try:
            assert dt.shape is not None, msg
            assert isinstance(dt.shape, int)
            return dt.shape
        except IndexError as e:
            raise Exception(
                f'Couldn\'t get a vector shape for {dt.huuid} due to empty shape'
            ) from e
