import typing as t
import uuid

import click
import numpy
import pandas
import redis
from redis.commands.json import JSON
from superduper import CFG
from superduper.backends.base.data_backend import BaseDataBackend
from superduper.base.query import Query, QueryPart
from superduper.base.schema import Schema


class RedisDataBackend(BaseDataBackend):
    """Redis data backend for SuperDuper.

    :param uri: The Redis URI.
    :param plugin: The plugin instance.
    :param flavour: The flavour of the backend.
    """

    def __init__(self, uri: str, plugin: t.Any, flavour: t.Optional[str] = None):
        super().__init__(uri, plugin, flavour)
        self.reconnect()

    def reconnect(self):
        """Redis connection."""
        self.conn = redis.Redis.from_url(self.uri)
        self.json_client = JSON(self.conn)

    def create_table_and_schema(self, identifier: str, schema: Schema, primary_id: str):
        """
        Create a table and schema in Redis.

        :param identifier: The name of the table.
        :param schema: The schema of the table.
        :param primary_id: The primary key field.
        """
        pass

    def drop_table(self, table: str):
        """Drop a table in Redis.

        :param table: The name of the table to drop.
        """
        keys = self.conn.keys(f"{table}:*")
        self.conn.delete(*keys)

    def drop(self, force: bool = False):
        """Drop the entire Redis DB.

        :param force: If True, drop the DB without confirmation.
        """
        if not force:
            if not click.confirm("Are you sure you want to drop the entire Redis DB?"):
                return

        self.conn.flushall()

    def list_tables(self) -> t.List[str]:
        """List all tables in Redis."""
        keys = self.conn.keys('*')
        return sorted(list(set(x.decode('utf-8').split(':')[0] for x in keys)))

    def get_table(self, identifier):
        """Get a table from Redis."""
        pass

    def check_output_dest(self, predict_id: str) -> bool:
        """Check if the output destination exists in Redis.

        :param predict_id: The ID of the prediction output.
        """
        tbl = CFG.output_prefix + predict_id
        return tbl in self.list_tables()

    def random_id(self) -> str:
        """Generate a random ID for Redis documents."""
        return str(uuid.uuid4()).replace('-', '')[:24]

    def primary_id(self, query: Query) -> str:
        """Get the primary ID field for a query."""
        return self.id_field

    def insert(self, table: str, documents: t.Sequence[t.Dict]) -> t.List[str]:
        """Insert documents into a Redis table.

        :param table: The name of the table.
        """
        if not documents:
            return []
        inserted_ids = []
        for doc in documents:
            if self.id_field not in doc:
                doc[self.id_field] = self.random_id()
            id = f'{table}:{doc[self.id_field]}'
            self.json_client.jsonset(id, '$', doc)
            inserted_ids.append(doc[self.id_field])
        return inserted_ids

    def replace(self, table: str, condition: t.Dict, r: t.Dict) -> t.List[str]:
        """Replace documents in a Redis table.

        :param table: The name of the table.
        :param condition: The condition to match documents.
        """
        df = self._get_table_df(table)

        query = []
        for k in condition:
            query.append(df[k] == condition[k])

        if len(query) > 1:
            query_combined = numpy.logical_and(*query)
        else:
            query_combined = query[0]

        ids = df[query_combined][self.id_field].tolist()

        for id in ids:
            s = self.json_client.jsonget(f'{table}:{id}')
            s.update(r)
            self.json_client.jsonset(f'{table}:{id}', '$', s)

        return ids

    def update(self, table: str, condition: t.Dict, key: str, value: t.Any):
        """Update documents in a Redis table.

        :param table: The name of the table.
        :param condition: The condition to match documents.
        :param key: The key to update.
        :param value: The new value for the key.
        """
        df = self._get_table_df(table)
        query = []
        for k in condition:
            query.append(df[k] == condition[k])

        if len(query) > 1:
            query_combined = numpy.logical_and(*query)
        else:
            query_combined = query[0]

        ids = df[query_combined][self.id_field].tolist()
        for id in ids:
            r = self.json_client.jsonget(f'{table}:{id}')
            r[key] = value
            self.json_client.jsonset(f'{table}:{id}', '$', r)
        return ids

    def delete(self, table: str, condition: t.Dict):
        """Delete documents from a Redis table.

        :param table: The name of the table.
        :param condition: The condition to match documents.
        """
        df = self._get_table_df(table)

        if df.empty:
            return

        query = []
        for k in condition:
            query.append(df[k] == condition[k])

        if len(query) > 1:
            query_combined = numpy.logical_and(*query)
        else:
            query_combined = query[0]

        ids = df[query_combined][self.id_field].tolist()

        for id in ids:
            self.conn.delete(f'{table}:{id}')

    def _build_pandas_df(self, query):

        q = self._get_table_df(query.table)
        pid = None
        predict_ids = (
            query.decomposition.outputs.args if query.decomposition.outputs else []
        )

        if q.empty:
            return q

        for part in query.parts:
            if isinstance(part, QueryPart) and part.name != 'outputs':
                args = []
                for a in part.args:
                    if isinstance(a, Query) and str(a).endswith('.primary_id'):
                        args.append(self.id_field)
                    elif isinstance(a, Query):
                        args.append(self._build_pandas_df(a))
                    else:
                        args.append(a)

                kwargs = {}
                for k, v in part.kwargs.items():
                    if isinstance(a, Query) and str(a).endswith('.primary_id'):
                        args.append(self.id_field)
                    elif isinstance(v, Query):
                        kwargs[k] = self._build_pandas_df(v)
                    else:
                        kwargs[k] = v

                if part.name == 'select' and len(args) == 0:
                    pass

                else:
                    if part.name == 'select' and predict_ids and args:
                        args.extend(
                            [
                                f'{CFG.output_prefix}{pid}'
                                for pid in predict_ids
                                if f'{CFG.output_prefix}{pid}' not in args
                            ]
                        )
                        args = list(set(args))

                    name = part.name
                    if name == 'limit':
                        name = 'head'

                    elif name == 'filter':
                        out = args[0]
                        if len(args) > 1:
                            for i in range(1, len(args)):
                                out = out & args[i]
                        q = q[out]

                    elif name == 'select':
                        assert all(isinstance(a, str) for a in args)
                        if not q.empty:
                            q = q[args]
                    else:
                        q = getattr(q, name)(*args, **kwargs)

            elif isinstance(part, QueryPart) and part.name == 'outputs':
                if pid is None:
                    pid = self.id_field

                for predict_id in part.args:
                    output_t = self._get_table_df(f"{CFG.output_prefix}{predict_id}")[
                        [f"{CFG.output_prefix}{predict_id}", "_source"]
                    ]
                    q = q.merge(output_t, left_on=pid, right_on='_source')

            elif isinstance(part, str):
                if part == 'primary_id':
                    if pid is None:
                        pid = self.id_field
                    part = pid
                q = q[part]
            else:
                raise ValueError(f'Unknown query part: {part}')

        return q

    def _get_table_df(self, table: str):
        keys = self.conn.keys(f"{table}:*")

        relevant_set = []
        for k in keys:
            relevant_set.append(self.json_client.jsonget(k))

        return pandas.DataFrame(relevant_set)

    def select(self, query: Query) -> t.List[t.Dict]:
        """Select documents from a Redis table.

        :param query: The query to execute.
        """
        df = self._build_pandas_df(query)
        return df.to_dict(orient='records')

    def missing_outputs(self, query: Query, predict_id: str) -> t.List[str]:
        """Select documents with missing outputs.

        :param query: The query to execute.
        :param predict_id: The ID of the prediction output.
        """
        pid = self.primary_id(query)
        df = self._build_pandas_df(query)
        output_df = self._get_table_df(f'{CFG.output_prefix + predict_id}')
        columns = output_df.columns
        columns = [c for c in columns if c != '"id"']
        output_df = output_df[columns]

        if output_df.empty:
            return df[pid].tolist()

        # TODO - map to pandas query
        joined_df = df.join(
            output_df, df[pid] == output_df['_source'], join_type="left"
        )
        return joined_df[joined_df['_source'].isnan()][self.id_field].tolist()

    def execute_native(self, query: str):
        """Execute a native Redis command (not-implemented)."""
        raise NotImplementedError
