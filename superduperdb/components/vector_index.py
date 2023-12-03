import dataclasses as dc
import typing as t

import numpy as np
from overrides import override

import superduperdb as s
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.component import Component
from superduperdb.components.encoder import Encoder
from superduperdb.components.listener import Listener
from superduperdb.ext.utils import str_shape
from superduperdb.misc.special_dicts import MongoStyleDict
from superduperdb.vector_search.base import VectorIndexMeasureType

if t.TYPE_CHECKING:
    pass


@dc.dataclass
class VectorIndex(Component):
    """
    A component carrying the information to apply a vector index to a ``DB`` instance

    :param identifier: Unique string identifier of index
    :param indexing_listener: Listener which is applied to created vectors
    :param compatible_listener: Listener which is applied to vectors to be compared
    :param measure: Measure to use for comparison
    :param version: version of this index
    :param metric_values: Metric values for this index
    """

    identifier: str
    indexing_listener: t.Union[Listener, str]
    compatible_listener: t.Union[None, Listener, str] = None
    measure: VectorIndexMeasureType = VectorIndexMeasureType.cosine
    version: t.Optional[int] = None
    metric_values: t.Optional[t.Dict] = dc.field(default_factory=dict)

    type_id: t.ClassVar[str] = 'vector_index'

    @override
    def post_create(self, db: Datalayer) -> None:
        super().post_create(db)
        if s.CFG.vector_search == s.CFG.data_backend:
            if (create := getattr(db.databackend, 'create_vector_index', None)) is None:
                msg = 'VectorIndex is not supported by the current database backend'
                raise ValueError(msg)
            create(self)

    @override
    def on_load(self, db: Datalayer) -> None:
        if isinstance(self.indexing_listener, str):
            self.indexing_listener = t.cast(
                Listener, db.load('listener', self.indexing_listener)
            )

        if isinstance(self.compatible_listener, str):
            self.compatible_listener = t.cast(
                Listener, db.load('listener', self.compatible_listener)
            )

    @property
    def child_components(self) -> t.Sequence[t.Tuple[str, str]]:
        out = [('indexing_listener', 'listener')]
        if self.compatible_listener is not None:
            out.append(('compatible_listener', 'listener'))
        return out

    def get_vector(
        self,
        like: Document,
        models: t.List[str],
        keys: t.List[str],
        db: t.Any = None,
        outputs: t.Optional[t.Dict] = None,
    ):
        document = MongoStyleDict(like.unpack())
        if outputs is not None:
            outputs = outputs or {}
            if '_outputs' not in document:
                document['_outputs'] = {}
            document['_outputs'].update(outputs)

            assert not isinstance(self.indexing_listener, str)
        available_keys = list(document.keys()) + ['_base']
        try:
            model_name, key = next(
                (m, k) for m, k in zip(models, keys) if k in available_keys
            )
        except StopIteration:
            raise Exception(
                f'Keys in provided {like} don\'t match'
                f'VectorIndex keys: {keys}, with model: {models}'
            )

        model_input = document
        if key == '_base' and key in document:
            model_input = document[key]
        elif key != '_base':
            model_input = document[key]

        model = db.models[model_name]
        return (
            model.predict(model_input, one=True),
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
        """Given a document, find the nearest results in this vector index, returned as
        two parallel lists of result IDs and scores

        :param like: The document to compare against
        :param db: The datastore to use
        :param outputs: An optional dictionary
        :param ids: A list of ids to match
        :param n: Number of items to return
        """

        models, keys = self.models_keys
        if len(models) != len(keys):
            raise ValueError(f'len(model={models}) != len(keys={keys})')
        within_ids = ids or ()

        if isinstance(like.content, dict) and id_field in like.content:
            return db.fast_vector_searchers[self.identifier].find_nearest_from_id(
                str(like[id_field]), within_ids=within_ids, limit=n
            )
        h = self.get_vector(
            like=like,
            models=models,
            keys=keys,
            db=db,
            outputs=outputs,
        )[0]

        return db.fast_vector_searchers[self.identifier].find_nearest_from_array(
            h, within_ids=within_ids, n=n
        )

    @property
    def models_keys(self) -> t.Tuple[t.List[str], t.List[str]]:
        """
        Return a list of model and keys for each listener
        """
        assert not isinstance(self.indexing_listener, str)
        assert not isinstance(self.compatible_listener, str)

        if self.compatible_listener:
            listeners = [self.indexing_listener, self.compatible_listener]
        else:
            listeners = [self.indexing_listener]

        models = [w.model.identifier for w in listeners]  # type: ignore[union-attr]
        keys = [w.key for w in listeners]
        return models, keys

    @property
    def dimensions(self) -> int:
        assert not isinstance(self.indexing_listener, str)
        assert not isinstance(self.indexing_listener.model, str)
        if shape := getattr(self.indexing_listener.model.encoder, 'shape', None):
            return shape[-1]
        raise ValueError('Couldn\'t get shape of model outputs from model encoder')


class EncodeArray:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, x):
        x = np.asarray(x)
        if x.dtype != self.dtype:
            raise TypeError(f'dtype was {x.dtype}, expected {self.dtype}')
        return memoryview(x).tobytes()


class DecodeArray:
    def __init__(self, dtype):
        self.dtype = dtype

    def __call__(self, bytes):
        return np.frombuffer(bytes, dtype=self.dtype).tolist()


def vector(shape):
    """
    Create an encoder for a vector (list of ints/ floats) of a given shape

    :param shape: The shape of the vector
    """
    return Encoder(
        identifier=f'vector[{str_shape(shape)}]',
        shape=shape,
        encoder=None,
        decoder=None,
    )


def sqlvector(shape):
    """
    Create an encoder for a vector (list of ints/ floats) of a given shape
    compatible with sql databases.

    :param shape: The shape of the vector
    """
    return Encoder(
        identifier=f'sqlvector[{str_shape(shape)}]',
        shape=shape,
        encoder=EncodeArray(dtype='float64'),
        decoder=DecodeArray(dtype='float64'),
    )
