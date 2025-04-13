import typing as t
import uuid
from collections import defaultdict

import pandas
from superduper import CFG, Document
from superduper.backends.base.query import (
    Query,
    applies_to,
    parse_query as _parse_query,
)
from superduper.base.cursor import SuperDuperCursor
from superduper.base.exceptions import DatabackendException
from superduper.components.datatype import _Encodable
from superduper.components.schema import Schema
from superduper.misc.special_dicts import SuperDuperFlatEncode

if t.TYPE_CHECKING:
    from superduper.base.datalayer import Datalayer


def parse_query(
    query, documents: t.Sequence[t.Dict] = (), db: t.Optional["Datalayer"] = None
):
    """Parse a string query into a query object.

    :param query: The query to parse.
    :param documents: The documents to query.
    :param db: The datalayer to use to execute the query.
    """
    return _parse_query(
        query=query,
        documents=list(documents),
        builder_cls=IbisQuery,
        db=db,
    )


def _load_keys_with_blob(output):
    if isinstance(output, SuperDuperFlatEncode):
        return output.load_keys_with_blob()
    elif isinstance(output, dict):
        return SuperDuperFlatEncode(output).load_keys_with_blob()
    return output


def _model_update_impl_flatten(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
):
    """Flatten the outputs and ids and update the model outputs in the database."""
    flattened_outputs = []
    flattened_ids = []
    for output, id in zip(outputs, ids):
        assert isinstance(output, (list, tuple)), "Expected list or tuple"
        for o in output:
            flattened_outputs.append(o)
            flattened_ids.append(id)

    return _model_update_impl(
        db=db,
        ids=flattened_ids,
        predict_id=predict_id,
        outputs=flattened_outputs,
    )


def _model_update_impl(
    db,
    ids: t.List[t.Any],
    predict_id: str,
    outputs: t.Sequence[t.Any],
):
    if not outputs:
        return

    documents = []
    for output, source_id in zip(outputs, ids):
        d = {
            "_source": str(source_id),
            f"{CFG.output_prefix}{predict_id}": (
                output.x if isinstance(output, _Encodable) else output
            ),
            "id": str(uuid.uuid4()),
        }
        documents.append(Document(d))
    return db[f"{CFG.output_prefix}{predict_id}"].insert(documents)


class IbisQuery(Query):
    """A query that can be executed on an Ibis database."""

    def __post_init__(self, db=None):
        super().__post_init__(db)
        self._primary_id = None
        self._base_table = None

    @property
    def base_table(self):
        """Return the base table."""
        if self._base_table is None:
            self._base_table = self.db.load('table', self.table)
        return self._base_table

    flavours: t.ClassVar[t.Dict[str, str]] = {
        "pre_like": r"^.*\.like\(.*\)\.select",
        "post_like": r"^.*\.([a-z]+)\(.*\)\.like(.*)$",
        "insert": r"^[^\(]+\.insert\(.*\)$",
        "filter": r"^[^\(]+\.filter\(.*\)$",
        "delete": r"^[^\(]+\.delete\(.*\)$",
        "select": r"^[^\(]+\.select\(.*\)$",
        "join": r"^.*\.join\(.*\)$",
        "anti_join": r"^[^\(]+\.anti_join\(.*\)$",
    }

    # Use to control the behavior in the class construction method within LeafMeta
    _dataclass_params: t.ClassVar[t.Dict[str, t.Any]] = {
        "eq": False,
        "order": False,
    }

    @property
    @applies_to("insert")
    def documents(self):
        """Return the documents."""
        return super().documents

    def _get_tables(self):
        table = self.db.load('table', self.table)
        out = {self.table: table}

        for part in self.parts:
            if isinstance(part, str):
                return out
            args = part[1]
            for a in args:
                if isinstance(a, IbisQuery):
                    out.update(a._get_tables())
            kwargs = part[2].values()
            for v in kwargs:
                if isinstance(v, IbisQuery):
                    out.update(v._get_tables())
        return out

    def _get_schema(self):
        fields = {}
        tables = self._get_tables()

        table_renamings = self.renamings({})
        if len(tables) == 1 and not table_renamings:
            table = self.db.load('table', self.table)
            return table.schema
        for identifier, c in tables.items():
            renamings = table_renamings.get(identifier, {})

            tmp = c.schema.fields
            to_update = dict(
                (renamings[k], v) if k in renamings else (k, v) for k, v in tmp.items()
            )
            fields.update(to_update)

        return Schema(f"_tmp:{self.table}", fields=fields, db=self.db)

    def renamings(self, r={}):
        """Return the renamings.

        :param r: Renamings.
        """
        for part in self.parts:
            if isinstance(part, str):
                continue
            if part[0] == "rename":
                r[self.table] = part[1][0]
            else:
                queries = list(part[1]) + list(part[2].values())
                for query in queries:
                    if isinstance(query, IbisQuery):
                        query.renamings(r)
        return r

    def _execute_select(self, parent):
        return self._execute(parent)

    def _execute_insert(self, parent):
        documents = self._prepare_documents()
        table = self.db.load('table', self.table)
        for r in documents:
            if table.primary_id not in r:
                pid = str(uuid.uuid4())
                r[table.primary_id] = pid
        ids = [r[table.primary_id] for r in documents]
        self.db.databackend.insert(self.table, raw_documents=documents)
        return ids

    def _create_table_if_not_exists(self):
        tables = self.db.databackend.list_tables_or_collections()
        if self.table in tables:
            return
        self.db.databackend.create_table_and_schema(
            self.table,
            self._get_schema(),
        )

    def _execute(self, parent, method="encode"):
        q = super()._execute(parent, method=method)
        try:
            output = q.execute()
        except Exception as e:
            raise DatabackendException(
                f"Error while executing ibis query {self}"
            ) from e

        assert isinstance(output, pandas.DataFrame)
        output = output.to_dict(orient="records")
        component_table = self.db.load('table', self.table)
        return SuperDuperCursor(
            raw_cursor=output,
            db=self.db,
            id_field=component_table.primary_id,
            schema=self._get_schema(),
        )

    @property
    def type(self):
        """Return the type of the query."""
        return defaultdict(
            lambda: "select",
            {
                "replace": "update",
                "delete": "delete",
                "filter": "select",
                "insert": "insert",
            },
        )[self.flavour]

    @property
    def primary_id(self):
        """Return the primary id."""
        return self.base_table.primary_id

    def model_update(
        self,
        ids: t.List[t.Any],
        predict_id: str,
        outputs: t.Sequence[t.Any],
        flatten: bool = False,
        **kwargs,
    ):
        """Update the model outputs in the database.

        :param ids: The ids of the inputs.
        :param predict_id: The predict id.
        :param outputs: The outputs.
        :param flatten: Whether to flatten the outputs.
        :param kwargs: Additional keyword arguments.
        """
        self.is_output_query = True
        self.updated_key = predict_id

        if not flatten:
            return _model_update_impl(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
            )
        else:
            return _model_update_impl_flatten(
                db=self.db,
                ids=ids,
                predict_id=predict_id,
                outputs=outputs,
            )

    def add_fold(self, fold: str):
        """Return a query that adds a fold.

        :param fold: The fold to add.
        """
        return self.filter(self._fold == fold)

    def select_using_ids(self, ids: t.Sequence[str]):
        """Return a query that selects using ids.

        :param ids: The ids to select.
        """
        filter_query = self.filter(getattr(self, self.primary_id).isin(ids))
        return filter_query

    @property
    def select_ids(self):
        """Return a query that selects ids."""
        return self.select(self.primary_id)

    def drop_outputs(self, predict_id: str):
        """Return a query that removes output corresponding to the predict id.

        :param predict_ids: The ids of the predictions to select.
        """
        return self.db.databackend.conn.drop_table(f"{CFG.output_prefix}{predict_id}")

    @applies_to("select")
    def outputs(self, *predict_ids):
        """Return a query that selects outputs.

        :param predict_ids: The predict ids.
        """
        for part in self.parts:
            if part[0] == "select":
                args = part[1]
                assert (
                    self.primary_id in args
                ), f"Primary id: `{self.primary_id}` not in select when using outputs"
        query = self
        attr = getattr(query, self.primary_id)
        for identifier in predict_ids:
            identifier = (
                identifier
                if identifier.startswith(CFG.output_prefix)
                else f"{CFG.output_prefix}{identifier}"
            )
            symbol_table = self.db[identifier]

            symbol_table = symbol_table.rename(
                # TODO: Check for folds
                {f"fold.{identifier}": "_fold", f"id.{identifier}": "id"}
            )
            query = query.join(symbol_table, symbol_table._source == attr)
        return query

    @applies_to("select", "join")
    def select_ids_of_missing_outputs(self, predict_id: str):
        """Return a query that selects ids of missing outputs.

        :param predict_id: The predict id.
        """
        from superduper.base.datalayer import Datalayer

        assert isinstance(self.db, Datalayer)

        output_table = self.db[f"{CFG.output_prefix}{predict_id}"]
        return self.anti_join(
            output_table,
            output_table._source == getattr(self, self.primary_id),
        )

    def select_single_id(self, id: str):
        """Return a query that selects a single id.

        :param id: The id to select.
        """
        filter_query = eval(f"table.{self.primary_id} == {id}")
        return self.filter(filter_query)

    @property
    def select_table(self):
        """Return a query that selects the table."""
        return self.db[self.table].select()

    def __call__(self, *args, **kwargs):
        """Add a method call to the query.

        :param args: The arguments to pass to the method.
        :param kwargs: The keyword arguments to pass to the method.
        """
        assert isinstance(self.parts[-1], str)
        # TODO: Move to _execute
        if (
            self.parts[-1] == "select"
            and not args
            and not self.table.startswith("<var:")
        ):
            # support table.select() without column args
            args = (IbisQuery(table=self.table, db=self.db),)
        return super().__call__(*args, **kwargs)

    def compile(self, db):
        """
        Compile `IbisQuery` to native ibis query format.

        :param db: Datalayer instance.
        """
        parent = self._get_parent()
        return super()._execute(parent).compile()


class _SQLDictIterable:
    def __init__(self, iterable):
        self.iterable = iter(iterable)

    def next(self):
        element = next(self.iterable)
        return dict(element)

    def __iter__(self):
        return self

    __next__ = next
