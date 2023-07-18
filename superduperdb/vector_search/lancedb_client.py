import dataclasses as dc
import os
import typing as t

import lancedb
import pandas as pd
import pyarrow as pa

import superduperdb as s
from superduperdb.misc.logger import logging
from superduperdb.vector_search.base import BaseVectorIndex
from .base import to_numpy
from superduperdb.vector_search.base import (
    VectorCollectionResult,
    VectorCollectionConfig,
    VectorCollectionItem,
)

_ID: str = 'id'
SEED_KEY: str = '__SEEDKEY__'


class LanceDBClient:
    def __init__(self, config: s.config.LanceDB) -> None:
        """
        Initialize the ``LanceDBClient``.

        :param config: Configuration object for vector search.
        """
        self.client = lancedb.connect(config.uri)
        self.config = config

    def get_table(self, table_name: str, measure: str = 'cosine') -> 'LanceTable':
        """
        Get a LanceTable from the ``LanceDBClient``.

        :param table_name: Name of the table.
        :param measure: Distance measure for vector search. Defaults to 'cosine'.
        """
        table = self.client.open_table(table_name)
        return LanceTable(client=self.client, table=table, measure=measure)

    def create_table(
        self,
        table_name: str,
        data: t.Optional[t.Sequence[t.Dict]] = None,
        schema: t.Optional[t.Dict] = None,
        measure: str = 'cosine',
    ) -> 'LanceTable':
        """
        Create a ``LanceTable``.

        :param table_name: Name of the table.
        :param data: Data to initialize the table with. Defaults to ``None``.
        :param schema: Schema of the table. Defaults to ``None``.
        :param measure: Distance measure for vector search. Defaults to 'cosine'.
        """
        if os.path.exists(os.path.join(self.config.uri, f'{table_name}.lance')):
            logging.debug(f'Table {table_name} already exists')
            table = self.client.open_table(table_name)
        else:
            table = self.client.create_table(table_name, data=data, schema=schema)
        return LanceTable(client=self.client, table=table, measure=measure)


VECTOR_FIELD_NAME = "vector"


@dc.dataclass
class LanceTable:
    client: lancedb.db.LanceDBConnection
    table: lancedb.table.LanceTable
    measure: str = "cosine"

    def get(self, identifier: str, limit: int = 1) -> t.List[t.Any]:
        """
        Get a vector from the ``LanceTable``.

        :param identifier: Identifier of the vector.
        """
        vector_df = self.table.search(f"id = '{identifier}'").limit(limit).to_df()
        vector = vector_df[VECTOR_FIELD_NAME]
        return vector

    def add(self, data: t.Sequence[VectorCollectionItem], upsert: bool = False) -> None:
        """
        Add vectors to the ``LanceTable``.

        :param data: Sequence of ``VectorCollectionItem`` objects.
        :param upsert: Whether to perform an upsert operation. Defaults to ``False``.
        """
        dict_data = [d.to_dict() for d in data]
        df = pd.DataFrame(dict_data)
        try:
            self.table.add(df)
        except ValueError:
            if upsert:
                self.client.create_table(self.table.name, df)
                return
            raise

    def find_nearest_from_id(
        self, identifier: t.Any, limit: int = 100, measure: t.Optional[str] = None
    ) -> t.List[VectorCollectionResult]:
        """
        Find nearest vectors to the vector with the given identifier.

        :param identifier: Identifier of the vector.
        :param limit: Maximum number of nearest vectors to return. Defaults to 100.
        :param measure: Distance measure for vector search. Defaults to ``None``.
        """
        vector = self.get(identifier, limit=limit)
        return self.find_nearest_from_array(vector, limit=limit, measure=measure)

    def find_nearest_from_array(
        self,
        array: t.Any,
        limit: int = 100,
        measure: t.Optional[str] = None,
        within_ids: t.Sequence = (),
    ) -> t.List[VectorCollectionResult]:
        """
        Find nearest vectors to the given array.

        :param array: Array representing the vector.
        :param limit: Maximum number of nearest vectors to return. Defaults to 100.
        :param measure: Distance measure for vector search. Defaults to ``None``.
        :param within_ids: ``List`` of identifiers to search within. Defaults to ().
        """
        if within_ids:
            raise NotImplementedError

        vectors_df = (
            self.table.search(to_numpy(array))
            .metric(measure if measure else self.measure)
            .limit(limit)
            .to_df()
        )
        ids = vectors_df[_ID].tolist()
        scores = vectors_df["score"].tolist()

        out = [
            VectorCollectionResult(id=id_, score=-distance)
            for id_, distance in zip(ids, scores)
            if id_ != SEED_KEY
        ]

        return out


class LanceVectorIndex(BaseVectorIndex):
    name: str = "lancedb"
    _ID: str = 'id'

    def __init__(self, config: s.config.LanceDB, measure: str = "cosine") -> None:
        """
        Initialize the ``LanceVectorIndex``.

        :param config: Configuration object for vector search.
        :param measure: Distance measure for vector search. Defaults to 'cosine'.
        """
        self.client = LanceDBClient(config)
        super().__init__(None, None, measure)

    def _create_schema(self, dimensions: int) -> pa.Schema:
        """
        Create the schema for the ``LanceTable``.

        :param dimensions: Number of ``dimensions`` of the vectors.
        """
        vector_type = pa.list_(pa.float32(), dimensions)
        return pa.schema(
            [pa.field(VECTOR_FIELD_NAME, vector_type), pa.field(self._ID, pa.string())]
        )

    def get_table(self, config: VectorCollectionConfig) -> 'LanceTable':
        """
        Get the ``LanceTable`` based on the ``VectorCollectionConfig``.

        :param config: Configuration object for vector collection.
        """
        dimensions = config.dimensions
        table = config.id
        measure = t.cast(str, config.measure)
        seed_data = list([{VECTOR_FIELD_NAME: [0] * dimensions, self._ID: SEED_KEY}])
        schema = self._create_schema(dimensions=dimensions)
        return self.client.create_table(
            table, schema=schema, measure=measure, data=seed_data
        )
