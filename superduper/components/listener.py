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
    metadata_fields: t.ClassVar[t.Dict[str, t.Type]] = {'output_table': Table}

    key: st.JSON
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Query] = None
    cdc_table: str = ''
    flatten: bool = False

    def postinit(self):
        """Post initialization method."""
        super().postinit()
        if not self.cdc_table and self.select:
            self.cdc_table = self.select.table
        if isinstance(self.key, tuple):
            self.key = list(self.key)

    @property
    def managed_tables(self):
        """Managed tables property."""
        return [self.outputs]

    @property
    def output_table(self):
        """Output table property."""
        t = Table(
            self.outputs,
            fields={self.outputs: self.model.datatype, '_source': 'str'},
        )
        t.status = 'running'
        return t

    @property
    def predict_id(self):
        """Predict ID property."""
        length_uuid = len(self.uuid)
        name_length = LENGTH_OUTPUT_NAME - length_uuid - 2
        name = self.identifier[:name_length]
        return f'{name}__{self.uuid}'

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

    def _check_signature(self):
        model_signature = self.model.signature

        msg = f'Invalid lookup key {self.key} for model signature {model_signature}'

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

    # The outputs parameter denotes the attribute of the component
    # to which the outputs of this job will be written.
    # This is needed if there are downstream jobs which depend on the outputs
    # of this job.
    # In this case `self.outputs` i.e. `CFG.output_prefix + self.predict_id`
    @trigger('apply', 'insert', 'update', requires='select', outputs='outputs')
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
            logging.info(
                f'[{self.huuid}] No documents to process for {self.huuid}, skipping'
            )
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

        logging.info(f"[{self.huuid}] Inserting {len(output_documents)} documents")
        result = self.db[self.outputs].insert(output_documents)
        logging.info(f"[{self.huuid}] Inserted {len(result)} documents")
        return result
