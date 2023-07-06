import dataclasses as dc

import lancedb
import pandas as pd

import superduperdb as s


class LanceDBClient:
    def __init__(self, config: s.config.LanceDB):
        self.client = lancedb.connect(config.uri)

    def get_table(self, table_name):
        table = self.client.open_table(table_name)
        return LanceTable(client=self, table=table)

    def create_table(self, table_name, data={}):
        tbl = self.client.create_table(table_name, data=data)
        return tbl


@dc.dataclass
class LanceTable:
    client: lancedb.db.LanceDBConnection
    table: lancedb.table.LanceTable

    def add(self, data, upsert: bool = False):
        df = pd.DataFrame(data)
        try:
            self.table.add(df)
        except ValueError:
            if upsert:
                self.client.create_table(self.table.name, df)
                return
            raise
