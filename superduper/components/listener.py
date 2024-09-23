import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.datalayer import Datalayer
from superduper.components.model import Mapping
from superduper.components.cdc import CDC
from superduper.base.annotations import trigger

from .model import Model, ModelInputType

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
    :param identifier: A string used to identify the listener and it's outputs.
    """

    type_id: t.ClassVar[str] = 'listener'

    key: ModelInputType
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Query] = None
    cdc_table: str = ''

    def __post_init__(self, db, artifacts):
        if not self.cdc_table and self.select:
            self.cdc_table = self.select.table
        deps = self.dependencies
        if deps:
            if not self.upstream:
                self.upstream = []
            for identifier, uuid in self.dependencies:
                self.upstream.append(f'&:component:listener:{identifier}:{uuid}')
        return super().__post_init__(db, artifacts)

    @property
    def predict_id(self):
        """Predict ID property."""
        return f'{self.identifier}__{self.uuid}'

    def pre_create(self, db: Datalayer) -> None:
        """Pre-create hook."""
        return super().pre_create(db)

    @property
    def mapping(self):
        """Mapping property."""
        return Mapping(self.key, signature=self.model.signature)

    @property
    def outputs(self):
        """Get reference to outputs of listener model."""
        return f'{CFG.output_prefix}{self.predict_id}'

    # TODO remove
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

    @override
    def post_create(self, db: "Datalayer") -> None:
        """Post-create hook.

        :param db: Data layer instance.
        """
        self.create_output_dest(db, self.predict_id, self.model)
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
            db.apply(output_table)

    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith(CFG.output_prefix):
                out.append(tuple(x[len(CFG.output_prefix) :].split('__')))
        return out

    @trigger('apply', 'insert', 'update', requires='select')
    def run(self, ids: t.Sequence[str] | None = None) -> t.List[str]:
        """Run the listener."""
        assert self.select is not None
        # Returns a list of ids where the outputs were inserted
        out = self.model.predict_in_db(
            X=self.key,
            predict_id=self.predict_id,
            select=self.select,
            ids=ids,
            **(self.predict_kwargs or {}),
        )
        return out

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        if self.select is not None:
            self.db[self.select.table].drop_outputs(self.predict_id)
