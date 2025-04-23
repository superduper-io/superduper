import dataclasses as dc
import typing as t

from superduper import CFG, Document, logging
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
LENGTH_OUTPUT_NAME = 50


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
        length_uuid = len(self.uuid)
        name_length = LENGTH_OUTPUT_NAME - length_uuid - 2
        name = self.identifier[:name_length]
        return f'{name}__{self.uuid}'

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

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

    @property
    def dependencies(self):
        """Get dependencies of this component."""
        tables = self.select.tables if self.select else []
        tables = [t for t in tables if t.startswith(CFG.output_prefix)]
        return [tuple(['Listener'] + list(t.split('__')[1:])) for t in tables]

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
        logging.info(f"[{self.huuid}] Running on '{self.cdc_table}'")

        self._check_signature()

        assert self.select is not None
        assert isinstance(self.db, Datalayer)

        if ids is None:
            logging.info(f'[{self.huuid}] No ids provided, using select {self.select}')
            ids = self.select.missing_outputs(self.predict_id)

        if not ids:
            logging.info(f'[{self.huuid}] No ids to process for {self.huuid}, skipping')
            return

        logging.info(f'[{self.huuid}] Processing {len(ids)} ids')
        if len(ids) <= 10:
            logging.info(f'[{self.huuid}] Processing ids: {ids}')
        else:
            logging.info(f'[{self.huuid}] Processing ids: {ids[:10]}...')

        documents = self.select.subset(ids)
        if not documents:
            return
        primary_id = self.select.primary_id.execute()
        output_primary_id = self.db[self.outputs].primary_id.execute()
        documents = [Document(d.unpack()) for d in documents]
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
            logging.info(
                f'[{self.huuid}] Flattened {len(outputs)} outputs into '
                f'{len(output_documents)} documents'
            )
        else:
            output_documents = [
                {
                    self.db.databackend.id_field: self.db.databackend.random_id(),
                    '_source': self.db.databackend.to_id(id),
                    output_primary_id: id,
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
