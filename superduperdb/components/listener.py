import dataclasses as dc
import typing as t

from overrides import override

from superduperdb import CFG, logging
from superduperdb.backends.base.query import Query
from superduperdb.base.document import _OUTPUTS_KEY
from superduperdb.base.enums import DBType
from superduperdb.components.model import Mapping
from superduperdb.misc.server import request_server

from ..jobs.job import Job
from .component import Component
from .model import Model, ModelInputType

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


class Listener(Component):
    """Listener component.

    Listener object which is used to process a column/key of a collection or table,
    and store the outputs.

    :param key: Key to be bound to the model.
    :param model: Model for processing data.
    :param select: Object for selecting which data is processed.
    :param active: Toggle to ``False`` to deactivate change data triggering.
    :param predict_kwargs: Keyword arguments to self.model.predict().
    :param identifier: A string used to identify the model.
    """

    key: ModelInputType
    model: Model
    select: Query
    active: bool = True
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''

    type_id: t.ClassVar[str] = 'listener'

    def __post_init__(self, db, artifacts):
        if self.identifier == '':
            self.identifier = self.uuid
        super().__post_init__(db, artifacts)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{_OUTPUTS_KEY}.{self.uuid}'

    @property
    def outputs_key(self):
        """Model outputs key."""
        # TODO: Conbine with outputs property after change the model output key
        # from 'output' to '_output.predict_id'
        if self.db.databackend.db_type == DBType.SQL:
            return 'output'
        else:
            return self.outputs

    @property
    def outputs_select(self):
        """Get query reference to model outputs."""
        if self.db.databackend.db_type == DBType.SQL:
            return self.db[self.outputs].select()

        else:
            model_update_kwargs = self.model.model_update_kwargs or {}
            if model_update_kwargs.get('document_embedded', True):
                collection_name = self.select.table
            else:
                collection_name = self.outputs
            return self.db[collection_name].find()

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.uuid, self.model)
        if self.select is not None and self.active and not db.server_mode:
            if CFG.cluster.cdc.uri:
                request_server(
                    service='cdc',
                    endpoint='listener/add',
                    args={'name': self.identifier},
                    type='get',
                )
            else:
                db.cdc.add(self)

    @classmethod
    def create_output_dest(cls, db: "Datalayer", uuid, model: Model):
        """
        Create output destination.

        :param db: Data layer instance.
        :param uuid: UUID of the listener.
        :param model: Model instance.
        """
        if model.datatype is None and model.output_schema is None:
            return
        output_table = db.databackend.create_output_dest(
            uuid,
            model.datatype,
            flatten=model.flatten,
        )
        if output_table is not None:
            db.add(output_table)

    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith('_outputs.'):
                listener_id = x.split('.')[1]
                out.append(listener_id)
        return out

    def depends_on(self, other: Component):
        """Check if the listener depends on another component.

        :param other: Another component.
        """
        if not isinstance(other, Listener):
            return False

        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())

        return any([x.startswith(f'_outputs.{other.uuid}') for x in all_])

    @property
    def predict_id(self):
        """Get predict ID."""
        return self.uuid

    @override
    def schedule_jobs(
        self,
        db: "Datalayer",
        dependencies: t.Sequence[Job] = (),
        overwrite: bool = False,
    ) -> t.Sequence[t.Any]:
        """Schedule jobs for the listener.

        :param db: Data layer instance to process.
        :param dependencies: A list of dependencies.
        :param overwrite: Overwrite the existing data.
        """
        if not self.active:
            return []
        assert not isinstance(self.model, str)

        dependencies_ids = []
        for predict_id in self.dependencies:
            try:
                upstream_listener = db.load(uuid=predict_id)
                upstream_model = upstream_listener.model
            except Exception:
                logging.warn(
                    f"Could not find the upstream listener with uuid {predict_id}"
                )
                continue
            jobs = self.db.metadata.show_jobs(upstream_model.identifier, 'model') or []
            job_ids = [job['job_id'] for job in jobs]
            dependencies_ids.extend(job_ids)

        dependencies = {*dependencies_ids, *dependencies}

        out = [
            self.model.predict_in_db_job(
                X=self.key,
                db=db,
                predict_id=self.uuid,
                select=self.select,
                dependencies=tuple(dependencies),
                overwrite=overwrite,
                **(self.predict_kwargs or {}),
            )
        ]
        return out

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        model_update_kwargs = self.model.model_update_kwargs or {}
        embedded = model_update_kwargs.get('document_embedded', True)
        self.db[self.select.table].drop_outputs(self.outputs, embedded=embedded)
