import dataclasses as dc
import os
import typing as t

import lancedb
import pandas as pd
import pyarrow as pa

from superduperdb import logging
from superduperdb.vector_search.base import (
    BaseVectorIndex,
    VectorCollectionConfig,
    VectorCollectionItem,
    VectorCollectionResult,
)

from .base import to_numpy

_ID: str = 'id'
SEED_KEY: str = '__SEEDKEY__'


class LanceDBClient:
    def __init__(self, uri: str) -> None:
        """
        Initialize the ``LanceDBClient``.

        :param config: Configuration object for vector search.
        """
        self.client = lancedb.connect(uri)
        self.uri = uri

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
        if os.path.exists(os.path.join(self.uri, f'{table_name}.lance')):
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

    def size(self) -> int:
        """
        Get the number of rows in ``LanceTable``.
        """
        # substract 1 for the seed vector
        return len(self.table) - 1

    def delete_from_ids(self, ids: t.Sequence[str]) -> None:
        """
        Delete vectors from the ``LanceTable``.

        :param ids: t.Sequence of identifiers.
        """
        to_remove = ", ".join(f"'{str(id)}'" for id in ids)
        self.table.delete(f"{_ID} IN ({to_remove})")

    def add(self, data: t.Sequence[VectorCollectionItem], upsert: bool = False) -> None:
        """
        Add vectors to the ``LanceTable``.

        :param data: t.Sequence of ``VectorCollectionItem`` objects.
        :param upsert: Whether to perform an upsert operation. Defaults to ``False``.
        """
        dict_data = []
        for d in data:
            dict_d = d.to_dict()
            vector = dict_d['vector']
            if hasattr(vector, 'numpy'):
                dict_d['vector'] = vector.numpy()
            dict_data.append(dict_d)

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
        :param within_ids: ``t.List`` of identifiers to search within. Defaults to ().
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

    def __init__(
        self,
        uri: str,
        measure: str = "cosine",
        client: t.Optional[LanceDBClient] = None,
    ) -> None:
        """
        Initialize the ``LanceVectorIndex``.

        :param uri: URI of the LanceDB database.
        :param measure: Distance measure for vector search. Defaults to 'cosine'.
        :param client: ``LanceDBClient`` instance. Defaults to ``None``.
        """
        if client:
            self.client = client
        else:
            self.client = LanceDBClient(uri=uri)
        self.measure = measure
        super().__init__(None, None, measure)

    def _create_schema(self, dimensions: int) -> pa.Schema:
        """
        Create the schema for the ``LanceTable``.

        :param dimensions: Number of ``dimensions`` of the vectors.
        """
        vector_type = pa.list_(pa.float32(), dimensions)
        return pa.schema(
            [pa.field(VECTOR_FIELD_NAME, vector_type), pa.field(_ID, pa.string())]
        )

    def get_table(
        self, config: VectorCollectionConfig, create: bool = False
    ) -> 'LanceTable':
        """
        Get the ``LanceTable`` based on the ``VectorCollectionConfig``.

        :param identifier: Identifier of the vector table.
        :param create: create the table if it does not exist. Defaults to ``False``.
        """
        try:
            return self.client.get_table(table_name=config.id)
        except FileNotFoundError:
            if create:
                return self.create_table(config=config)
            raise

    def create_table(self, config: VectorCollectionConfig) -> 'LanceTable':
        """
        Create the ``LanceTable`` based on the ``VectorCollectionConfig``.

        :param config: Configuration object for vector collection.
        """
        dimensions = config.dimensions
        table = config.id
        measure = t.cast(str, config.measure)
        seed_data = list([{VECTOR_FIELD_NAME: [0] * dimensions, _ID: SEED_KEY}])
        schema = self._create_schema(dimensions=dimensions)
        return self.client.create_table(
            table, schema=schema, measure=measure, data=seed_data
        )
