import dataclasses as dc
import typing as t

from overrides import override

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base.annotations import trigger
from superduper.base.datalayer import Datalayer
from superduper.components.cdc import CDC
from superduper.components.model import Mapping
from superduper.components.table import Table

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
    :param output_table: Table to store the outputs.
    """

    type_id: t.ClassVar[str] = 'listener'

    key: ModelInputType
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Query] = None
    cdc_table: str = ''
    output_table: t.Optional[Table] = None

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

    def _pre_create(self, db: Datalayer, startup_cache: t.Dict):
        """Pre-create hook."""
        if self.select is None:
            return
        if not db.cfg.auto_schema:
            return
        if self.output_table is not None:
            return

        from superduper import Schema
        if self.model.datatype is not None:
            schema = Schema(
                f'_schema/{self.outputs}',
                fields={self.outputs: self.model.datatype}
            )
        elif self.model.output_schema is not None:
            schema = self.model.output_schema
        else:
            if self.model.example is not None:
                input = self.model.example
            else:
                try:
                    if self.dependencies:
                        try:
                            r = next(self.select.limit(1).execute())
                        except StopIteration as e:
                            try:
                                if not self.cdc_table.startswith(CFG.output_prefix):
                                    r = next(db[self.select.table].select().limit(1).execute())
                                    r = {**r, **startup_cache}
                                else:
                                    r = startup_cache
                            except StopIteration as e:
                                raise Exception(
                                    f'Couldn\'t retrieve outputs to determine schema; {self.cdc_table} returned 0 results.'
                                ) from e
                    else:
                        r = next(self.select.limit(1).execute())
                except StopIteration as e:
                    raise Exception(
                        f'Couldn\'t retrieve outputs to determine schema; {self.select} returned 0 results.'
                    ) from e

                mapping = Mapping(self.key, self.model.signature)
                input = mapping(r)

            if self.model.signature == 'singleton':
                prediction = self.model.predict(input)
            elif self.model.signature == '*args':
                prediction = self.model.predict(*input)
            elif self.model.signature == '**kwargs':
                prediction = self.model.predict(**input)
            else:
                assert self.model.signature == '*args,**kwargs'
                prediction = self.model.predict(*input[0], **input[1])

            if self.model.flatten:
                prediction = prediction[0]

            startup_cache[self.outputs] = prediction
            schema = db.infer_schema({'data': prediction}, identifier=self.outputs + '/schema')
            datatype = schema.fields['data']
            self.model.datatype = datatype
            schema=Schema(f'_schema/{self.outputs}', fields={'_source': 'ID', 'id': 'ID', self.outputs: datatype})
        
        self.output_table = Table(self.outputs, schema=schema)

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
        return self.db[self.outputs].select()

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
