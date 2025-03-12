import dataclasses as dc
import typing as t

from superduper import CFG
from superduper.base.annotations import trigger
from superduper.base.datalayer import Datalayer
from superduper.base.query import Query
from superduper.components.cdc import CDC
from superduper.components.table import Table
from superduper.misc import typing as st

from .model import Model

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


class Listener(CDC):
    """Listener component.

    Listener object which is used to process a column/key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to the model.
    :param model: Model for processing data.
    :param predict_kwargs: Keyword arguments to self.model.predict().
    :param select: Query to "listen" for input on.
    :param flatten: Flatten the output into separate records if ``True``.
    """

    breaks: t.ClassVar[t.Sequence[str]] = ('model', 'key', 'select')

    key: st.JSON
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Query] = None
    cdc_table: str = ''
    flatten: bool = False

    def postinit(self):
        """Post initialization method."""
        if not self.cdc_table and self.select:
            self.cdc_table = self.select.table
        if isinstance(self.key, tuple):
            self.key = list(self.key)

        super().postinit()

    @property
    def output_table(self):
        return Table(
            self.outputs, fields={self.outputs: self.model.datatype, '_source': 'str'}
        )

    def _get_metadata(self):
        r = super()._get_metadata()
        return {**r, 'output_table': self.output_table}

    @property
    def predict_id(self):
        """Predict ID property."""
        return f'{self.identifier}__{self.uuid}'

    # TODO deprecate this
    @staticmethod
    def _complete_key(key, db, listener_uuids=()):
        if isinstance(key, str) and key.startswith(CFG.output_prefix):
            if len(key[len(CFG.output_prefix) :].split('__')) == 2:
                return key
            identifier_and_sub_key = key[len(CFG.output_prefix) :].split('.', 1)
            if len(identifier_and_sub_key) == 2:
                identifier, sub_key = identifier_and_sub_key
            else:
                identifier = identifier_and_sub_key[0]
                sub_key = ''

            key = CFG.output_prefix + identifier
            try:
                uuid = listener_uuids[identifier]
            except KeyError:
                try:
                    uuid = db.show('listener', identifier, -1)['uuid']
                except FileNotFoundError:
                    raise Exception(
                        'Couldn\'t complete `Listener` key '
                        f'based on ellipsis {key}__????????????????. '
                        'Please specify using upstream_listener.outputs'
                    )

            complete_key = key + '__' + uuid
            if sub_key:
                complete_key += '.' + sub_key
            return complete_key
        elif isinstance(key, str):
            return key
        elif isinstance(key, list):
            return [Listener._complete_key(k, db, listener_uuids) for k in key]
        elif isinstance(key, tuple):
            return tuple([Listener._complete_key(k, db, listener_uuids) for k in key])
        elif isinstance(key, dict):
            return {
                Listener._complete_key(k, db, listener_uuids): v for k, v in key.items()
            }
        raise Exception(f'Invalid key type: {type(key)}')

    def _auto_fill_data(self, db: Datalayer):
        listener_keys = [k for k in db.startup_cache if k.startswith(CFG.output_prefix)]
        listener_predict_ids = [k[len(CFG.output_prefix) :] for k in listener_keys]
        lookup: t.Dict = dict(tuple(x.split('__')) for x in listener_predict_ids)
        assert self.select is not None
        self.select = self.select.complete_uuids(db, listener_uuids=lookup)
        if CFG.output_prefix in str(self.key):
            self.key = self._complete_key(self.key, db, listener_uuids=lookup)

        if self.cdc_table.startswith(CFG.output_prefix):
            self.cdc_table = self.select.table

    def _pre_create(self, db: Datalayer, startup_cache: t.Dict = {}):
        """Pre-create hook."""
        if self.select is None:
            return

        # TODO deprecate this
        self._auto_fill_data(db)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

    # TODO deprecate
    @property
    def dependencies(self):
        """Listener model dependencies."""
        # args, kwargs = self.mapping.mapping
        key = self.key[:]
        if isinstance(key, str):
            all_ = [key]
        elif isinstance(key, (list, tuple)) and isinstance(key[0], str):
            all_ = list(key)
        else:
            all_ = list(key[0]) + list(key[1].values())

        # all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith(CFG.output_prefix):
                out.append(tuple(x[len(CFG.output_prefix) :].split('.')[0].split('__')))
        return out

    def _check_signature(self):
        model_signature = self.model.signature

        msg = 'Invalid lookup key {self.key} for model signature {model_signature}'

        if model_signature == 'singleton':
            assert isinstance(self.key, str) or self.key is None, msg
        elif model_signature == '*args':
            assert isinstance(self.key, (list, tuple)), msg
            assert all(isinstance(x, str) for x in self.key), msg
        elif model_signature == '**kwargs':
            assert isinstance(self.key, dict), msg
        elif model_signature == '*args,**kwargs':
            assert isinstance(self.key, (list, tuple)), msg
            assert isinstance(self.key[0], (list, tuple)), msg
            assert all(isinstance(x, str) for x in self.key[0]), msg
            assert isinstance(self.key[1], dict), msg
            assert all(isinstance(x, str) for x in self.key[1].values()), msg
        else:
            raise ValueError(f'Invalid signature: {model_signature}')

    @trigger('apply', 'insert', 'update', requires='select')
    def run(self, ids: t.List[str] | None = None):

        self._check_signature()

        assert self.select is not None
        assert isinstance(self.db, Datalayer)

        if ids is None:
            ids = self.select.missing_outputs(self.predict_id)

        if not ids:
            return

        documents = self.select.subset(ids)
        if not documents:
            return
        primary_id = self.select.primary_id.execute()
        ids = [r[primary_id] for r in documents]

        inputs = self.model._map_inputs(self.model.signature, documents, self.key)
        outputs = self.model.predict_batches(inputs)
        if self.flatten:
            output_documents = [
                {
                    self.db.databackend.id_field: self.db.databackend.random_id(),
                    '_source': self.db.databackend.to_id(id),
                    self.outputs: sub_output,
                }
                for id, output in zip(ids, outputs)
                for sub_output in output
            ]
        else:
            output_documents = [
                {
                    self.db.databackend.id_field: self.db.databackend.random_id(),
                    '_source': self.db.databackend.to_id(id),
                    self.outputs: output,
                }
                for id, output in zip(ids, outputs)
            ]
        return self.db[self.outputs].insert(output_documents)

    def cleanup(self):
        """Clean up when the listener is deleted."""
        super().cleanup()
        if self.select is not None:
            self.db.databackend.drop_table(self.outputs)
