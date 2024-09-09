import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.datalayer import Datalayer
from superduper.base.event import Event
from superduper.components.model import Mapping
from superduper.components.trigger import Trigger
from superduper.jobs.annotations import trigger
from superduper.misc.server import request_server

from ..jobs.job import Job
from .model import Model, ModelInputType

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


class Listener(Trigger):
    """Listener component.

    Listener object which is used to process a column/key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to the model.
    :param model: Model for processing data.
    :param predict_kwargs: Keyword arguments to self.model.predict().
    :param identifier: A string used to identify the listener and it's outputs.
    """

    key: ModelInputType
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, db, artifacts):
        deps = self.dependencies
        if deps:
            if not self.upstream:
                self.upstream = []
            for identifier, uuid in self.dependencies:
                self.upstream.append(f'&:component:listener:{identifier}:{uuid}')
        return super().__post_init__(db, artifacts)

    @property
    def predict_id(self):
        return f'{self.identifier}__{self.uuid}'

    def pre_create(self, db: Datalayer) -> None:
        return super().pre_create(db)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    # TODO - do we need the outputs-prefix?
    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

    @property
    def outputs_key(self):
        """Model outputs key."""
        logging.warn(
            (
                "listener.outputs_key is deprecated and will be removed"
                "in a future release. Please use listener.outputs instead."
            )
        )
        return self.outputs

    @property
    def outputs_select(self):
        """Get select statement for outputs."""
        return self.db[self.select.table].select().outputs(self.predict_id)

    # TODO do we need this?
    @property
    def cdc_table(self):
        """Get table for cdc."""
        return self.select.table_or_collection.identifier

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.predict_id, self.model)
        if self.select is not None:
            logging.info('Requesting listener setup on CDC service')
            if CFG.cluster.cdc.uri and not self.dependencies:
                logging.info('Sending request to add listener')
                request_server(
                    service='cdc',
                    endpoint='component/add',
                    args={'name': self.identifier, 'type_id': self.type_id},
                    type='get',
                )
            else:
                logging.info(
                    'Skipping listener setup on CDC service since no URI is set'
                )
        else:
            logging.info('Skipping listener setup on CDC service')
        super().post_create(db)

    @classmethod
    def create_output_dest(cls, db: "Datalayer", predict_id, model: Model):
        """
        Create output destination.

        :param db: Data layer instance.
        :param uuid: UUID of the listener.
        :param model: Model instance.
        """
        if model.datatype is None and model.output_schema is None:
            return
        # TODO make this universal over databackends
        # not special e.g. MongoDB vs. Ibis creating a table or not
        output_table = db.databackend.create_output_dest(
            predict_id,
            model.datatype,
            flatten=model.flatten,
        )
        if output_table is not None:
            db.add(output_table)

    # TODO rename
    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith(CFG.output_prefix):
                out.append(tuple(x[len(CFG.output_prefix):].split('__')))
        return out

    def trigger_ids(self, query: Query, primary_ids: t.Sequence):
        """Get trigger IDs.

        Only the ids returned by this function will trigger the listener.

        :param query: Query object.
        :param primary_ids: Primary IDs.
        """
        conditions = [
            # trigger by main table
            self.select and self.select.table == query.table,
            # trigger by output table
            query.table in self.key and query.table != self.outputs,
        ]
        if not any(conditions):
            return []

        if self.select is None:
            return []

        if self.select.table == query.table:
            trigger_ids = list(primary_ids)

        else:
            trigger_ids = [
                doc['_source'] for doc in query.documents if '_source' in doc
            ]

        return self.db.databackend.check_ready_ids(
            self.select, self._ready_keys, trigger_ids
        )

    @property
    def _ready_keys(self):
        keys = self.key

        if isinstance(self.key, str):
            keys = [self.key]
        elif isinstance(self.key, dict):
            keys = list(self.key.keys())

        # Support multiple levels of nesting
        clean_keys = []
        for key in keys:
            if key.startswith(CFG.output_prefix):
                key = CFG.output_prefix + key[len(CFG.output_prefix) :].split(".")[0]
            else:
                key = key.split(".")[0]

            clean_keys.append(key)

        return clean_keys

    @trigger('apply', 'insert', 'update', requires='select')
    def run(self, ids: t.Sequence[str] | None):
        return self.model.predict_in_db(
            X=self.key,
            predict_id=self.predict_id,
            select=self.select,
            ids=ids,
            **(self.predict_kwargs or {}),
        )

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        if self.select is not None:
            self.db[self.select.table].drop_outputs(self.predict_id)
