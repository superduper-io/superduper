import lancedb
import pandas as pd

import superduperdb as s


class LanceDBClient:
    def __init__(self, config: s.config.LanceDB):
        self.client = lancedb.connect(config.uri)

    def get_table(self, table_name):
        tbl = self.client.open_table(table_name)
        return LanceTable(tbl)

    def create_table(self, table_name, data={}):
        tbl = self.client.create_table(table_name, data=data)
        return tbl


class LanceTable:
    def __init__(self, table):
        self._table = table

    @property
    def table(
        self,
    ):
        return self._table

    def add(self, data):
        df = pd.DataFrame(data)
        self.table.add(df)
