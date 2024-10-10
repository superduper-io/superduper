import dataclasses as dc
import typing as t
from copy import copy

from superduper import CFG, logging
from superduper.backends.base.query import Query
from superduper.base import exceptions
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
    :param flatten: Flatten the output into separate records if ``True``.
    """

    type_id: t.ClassVar[str] = 'listener'

    key: ModelInputType
    model: Model
    predict_kwargs: t.Optional[t.Dict] = dc.field(default_factory=dict)
    select: t.Optional[Query] = None
    cdc_table: str = ''
    output_table: t.Optional[Table] = None
    flatten: bool = False

    def __post_init__(self, db, artifacts):
        if not self.cdc_table and self.select:
            self.cdc_table = self.select.table
        self._set_upstream()

        return super().__post_init__(db, artifacts)

    def dict(self, metadata: bool = True, defaults: bool = True) -> t.Dict:
        """Convert to dictionary."""
        out = super().dict(metadata=metadata, defaults=defaults)
        if not metadata:
            try:
                del out['output_table']
            except KeyError:
                logging.warn('output_table not found in listener.dict()')
                pass
        return out

    def _set_upstream(self):
        deps = self.dependencies
        if deps:
            if not self.upstream:
                self.upstream = []
            try:
                it = 0
                for dep in deps:
                    identifier, uuid = dep
                    self.upstream.append(f'&:component:listener:{identifier}:{uuid}')
                    it += 1
            except ValueError as e:
                if 'not enough values' in str(e):
                    logging.warn(
                        'Deferring dependencies to pre_create based on '
                        f'dependency {deps[it]}'
                    )

    @property
    def predict_id(self):
        """Predict ID property."""
        return f'{self.identifier}__{self.uuid}'

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

    def _get_sample_input(self, db: Datalayer):
        msg = (
            'Couldn\'t retrieve outputs to determine schema; '
            '{table} returned 0 results.'
        )
        errors = (
            StopIteration,
            KeyError,
            FileNotFoundError,
            exceptions.TableNotFoundError,
        )
        if self.model.example is not None:
            input = self.model.example
        else:
            if self.dependencies:
                try:
                    r = next(self.select.limit(1).execute())
                except errors:
                    try:
                        if not self.cdc_table.startswith(CFG.output_prefix):
                            try:
                                r = next(
                                    db[self.select.table].select().limit(1).execute()
                                )
                            except errors:
                                # Note: This is added for sql databases,
                                # since they return error if key not found
                                # unlike mongodb
                                r = {}
                            r = {**r, **db.startup_cache}
                        else:
                            r = copy(db.startup_cache)
                    except Exception as e:
                        raise Exception(msg.format(table=self.cdc_table)) from e
            else:
                try:
                    r = next(self.select.limit(1).execute())
                except (StopIteration, KeyError, FileNotFoundError) as e:
                    raise Exception(msg.format(table=self.select)) from e
            mapping = Mapping(self.key, self.model.signature)
            input = mapping(r)
        return input

    def _determine_table_and_schema(self, db: Datalayer):
        from superduper import Schema

        if self.model.datatype is not None:
            schema = Schema(
                f'_schema/{self.outputs}',
                fields={'_source': 'ID', self.outputs: self.model.datatype},
            )
        elif self.model.output_schema is not None:
            schema = self.model.output_schema
        else:
            input = self._get_sample_input(db)

            if self.model.signature == 'singleton':
                prediction = self.model.predict(input)
            elif self.model.signature == '*args':
                prediction = self.model.predict(*input)
            elif self.model.signature == '**kwargs':
                prediction = self.model.predict(**input)
            else:
                assert self.model.signature == '*args,**kwargs'
                prediction = self.model.predict(*input[0], **input[1])

            if self.flatten:
                prediction = prediction[0]

            db.startup_cache[self.outputs] = prediction
            schema = db.infer_schema(
                {'data': prediction}, identifier=self.outputs + '/schema'
            )
            datatype = schema.fields['data']
            self.model.datatype = datatype
            schema = Schema(
                f'_schema/{self.outputs}',
                fields={'_source': 'ID', 'id': 'ID', self.outputs: datatype},
            )

        self.output_table = Table(self.outputs, schema=schema)

    def _pre_create(self, db: Datalayer, startup_cache: t.Dict = {}):
        """Pre-create hook."""
        if self.select is None:
            return

        if not db.cfg.auto_schema:
            db.startup_cache[self.outputs] = None
            return

        self._auto_fill_data(db)

        if self.output_table is not None:
            return

        self._determine_table_and_schema(db)

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

    @property
    def dependencies(self):
        """Listener model dependencies."""
        args, kwargs = self.mapping.mapping
        all_ = list(args) + list(kwargs.values())
        out = []
        for x in all_:
            if x.startswith(CFG.output_prefix):
                out.append(tuple(x[len(CFG.output_prefix) :].split('.')[0].split('__')))
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
            flatten=self.flatten,
            **(self.predict_kwargs or {}),
        )
        return out

    def cleanup(self, db: "Datalayer") -> None:
        """Clean up when the listener is deleted.

        :param db: Data layer instance to process.
        """
        if self.select is not None:
            db[self.select.table].drop_outputs(self.predict_id)
