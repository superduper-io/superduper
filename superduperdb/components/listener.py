import dataclasses as dc
import typing as t

from overrides import override

from superduperdb import CFG
from superduperdb.backends.base.query import CompoundSelect
from superduperdb.base.document import _OUTPUTS_KEY
from superduperdb.base.serializable import Variable
from superduperdb.components.model import Mapping
from superduperdb.misc.annotations import public_api
from superduperdb.misc.server import request_server
from superduperdb.rest.utils import parse_query

from ..jobs.job import Job
from .component import Component, ComponentTuple
from .model import Model, ModelInputType

if t.TYPE_CHECKING:
    from superduperdb.base.datalayer import Datalayer


SELECT_TEMPLATE = {'documents': [], 'query': '<collection_name>.find()'}


@public_api(stability='stable')
@dc.dataclass(kw_only=True)
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

    __doc__ = __doc__.format(component_parameters=Component.__doc__)

    ui_schema: t.ClassVar[t.List[t.Dict]] = [
        {'name': 'identifier', 'type': 'str', 'default': ''},
        {'name': 'key', 'type': 'json'},
        {'name': 'model', 'type': 'component/model'},
        {'name': 'select', 'type': 'json', 'default': SELECT_TEMPLATE},
        {'name': 'active', 'type': 'bool', 'default': True},
        {'name': 'predict_kwargs', 'type': 'json', 'default': {}},
    ]

    key: ModelInputType
    model: Model
    select: CompoundSelect
    active: bool = True
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    identifier: str = ''

    type_id: t.ClassVar[str] = 'listener'

    @classmethod
    def handle_integration(cls, kwargs):
        """Method to handle integration.

        :param kwargs: Integration keyword arguments.
        """
        if 'select' in kwargs and isinstance(kwargs['select'], dict):
            kwargs['select'] = parse_query(
                query=kwargs['select']['query'],
                documents=kwargs['select']['documents'],
            )
        return kwargs

    def __post_init__(self, artifacts):
        if self.identifier == '':
            if self.model.version is not None:
                self.identifier = (
                    f'{self.id_key}::{self.model.identifier}::{self.model.version}'
                )
            else:
                self.identifier = Variable(
                    f'{self.id_key}::{self.model.identifier}::{"{model_version}"}',
                    setter_callback=lambda db, value, kwargs: value.format(
                        model_version=self.model.version,
                    ),
                )
        super().__post_init__(artifacts)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        if self.model.version is not None:
            return f'{_OUTPUTS_KEY}.{self.identifier}::{self.version}'
        else:

            def _callback(db, value, kwargs):
                return value.format(version=self.version)

            return Variable(
                f'{_OUTPUTS_KEY}.{self.identifier}::{"{version}"}',
                setter_callback=_callback,
            )

    @property
    def outputs_select(self):
        """Get query reference to model outputs."""
        if self.select.DB_TYPE == "SQL":
            return self.select.table_or_collection.outputs(self.predict_id)

        else:
            from superduperdb.backends.mongodb.query import Collection

            model_update_kwargs = self.model.model_update_kwargs or {}
            if model_update_kwargs.get('document_embedded', True):
                collection_name = self.select.table_or_collection.identifier
            else:
                collection_name = self.outputs

            return Collection(collection_name).find()

    @property
    def outputs_key(self):
        """Model outputs key."""
        if self.select.DB_TYPE == "SQL":
            return 'output'
        else:
            return self.outputs

    @override
    def pre_create(self, db: "Datalayer") -> None:
        """Pre-create hook.

        :param db: Data layer instance.
        """
        if self.select is not None and self.select.variables:
            self.select = t.cast(CompoundSelect, self.select.set_variables(db))

        # This logic intended to format the version of upstream `Listener`
        # instances into `self.key`, so that the `Listener` knows exactly where
        # to find the `_outputs` subfield: `_outputs.<identifier>.<version>`
        # of a `Listener`.
        self.key = self._set_key(db, self.key)

    @staticmethod
    def _set_key(db, key, **kwargs):
        if isinstance(key, Variable):
            return key.set(db, **kwargs)
        elif isinstance(key, (list, tuple)):
            return [Listener._set_key(db, x, **kwargs) for x in key]
        elif isinstance(key, dict):
            return {k: Listener._set_key(db, v, **kwargs) for k, v in key.items()}
        return key

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.predict_id, self.model)
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
    def create_output_dest(cls, db: "Datalayer", predict_id, model: Model):
        """
        Create output destination.

        :param db: Data layer instance.
        :param predict_id: Predict ID.
        :param model: Model instance.
        """
        if model.datatype is None:
            return
        output_table = db.databackend.create_output_dest(
            predict_id,
            model.datatype,
            flatten=model.flatten,
        )
        if output_table is not None:
            db.add(output_table)

    @property
    def dependencies(self) -> t.List[ComponentTuple]:
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith('_outputs.'):
                parts = x.split('.')[1].split('::')
                listener_id, version = '::'.join(parts[:-1]), int(parts[-1])
                out.append(ComponentTuple('listener', listener_id, version))
        return out

    @property
    def predict_id(self):
        """Get predict ID."""
        return f'{self.identifier}::{self.version}'

    @classmethod
    def from_predict_id(cls, db: "Datalayer", predict_id) -> 'Listener':
        """
        Split predict ID.

        :param db: Data layer instance.
        :param predict_id: Predict ID.
        """
        identifier, version = predict_id.rsplit('::', 1)
        return t.cast(Listener, db.load('listener', identifier, version=int(version)))

    @property
    def id_key(self) -> str:
        """Get identifier key."""

        def _id_key(key) -> str:
            if isinstance(key, str):
                if key.startswith('_outputs.'):
                    return key.split('.')[1]
                else:
                    return key
            elif isinstance(key, (tuple, list)):
                return ','.join([_id_key(k) for k in key])
            elif isinstance(key, dict):
                return ','.join([_id_key(k) for k in key.values()])
            else:
                raise TypeError('Type of key is not valid')

        return _id_key(self.key)

    def depends_on(self, other: Component):
        """Check if the listener depends on another component.

        :param other: Another component.
        """
        if not isinstance(other, Listener):
            return False

        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())

        return any([x.startswith(f'_outputs.{other.identifier}') for x in all_])

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
        """
        if not self.active:
            return []
        assert not isinstance(self.model, str)

        out = [
            self.model.predict_in_db_job(
                X=self.key,
                db=db,
                predict_id=self.predict_id,
                select=self.select.copy(),
                dependencies=dependencies,
                overwrite=overwrite,
                **(self.predict_kwargs or {}),
            )
        ]
        return out

    def cleanup(self, database: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param database: Data layer instance to process.
        """
        # TODO - this doesn't seem to do anything
        if (cleanup := getattr(self.select, 'model_cleanup', None)) is not None:
            assert not isinstance(self.model, str)
            cleanup(database, model=self.model.identifier, key=self.key)
