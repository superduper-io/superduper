from functools import wraps
import os
import re

import ibis
import pandas
import snowflake.connector

from superduper import logging
from superduper_ibis.data_backend import IbisDataBackend
from snowflake.snowpark import Session


class SnowflakeDataBackend(IbisDataBackend):

    @wraps(IbisDataBackend.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.snowpark = self._get_snowpark_session(self.uri)

    @staticmethod
    def _get_snowpark_session(uri):
        logging.info('Creating Snowpark session')
        if uri == 'snowflake://':
            connection_parameters = dict(
                host=os.environ['SNOWFLAKE_HOST'],
                port=int(os.environ['SNOWFLAKE_PORT']),
                account=os.environ['SNOWFLAKE_ACCOUNT'],
                authenticator='oauth',
                token=open('/snowflake/session/token').read(),
                warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
                database=os.environ['SNOWFLAKE_DATABASE'],
                schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
            )
        else:
            if '?warehouse=' not in uri:
                match = re.match(
                    '^snowflake:\/\/(.*):(.*)\@(.*)\/(.*)\/(.*)$', uri
                )
                user, password, account, database, schema = match.groups()
                warehouse = None
            else:
                match = re.match(
                    'snowflake://(.*):(.*)@(.*)/(.*)/(.*)?warehouse=(.*)^'
                )
                password, user, account, database, schema, warehouse = match.groups()

            connection_parameters = dict(
                user=user,
                password=password,
                account=account,
                warehouse=warehouse,
                database=database,
                schema=schema,
            )

        return Session.builder.configs(connection_parameters).create()

    @staticmethod
    def _do_connection_callback(uri):
        logging.info('Using env variables and OAuth to connect!')
        return snowflake.connector.connect(
            host=os.environ['SNOWFLAKE_HOST'],
            port=int(os.environ['SNOWFLAKE_PORT']),
            account=os.environ['SNOWFLAKE_ACCOUNT'],
            authenticator='oauth',
            token=open('/snowflake/session/token').read(),
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
        )

    def _connection_callback(self, uri):
        if uri != 'snowflake://':
            return IbisDataBackend._connection_callback(uri)
        return ibis.snowflake.from_connection(self._do_connection_callback(uri)), 'snowflake', False

    def insert(self, table_name, raw_documents):
        df = pandas.DataFrame(raw_documents)
        columns = df.columns
        rows = list(df.itertuples(index=False, name=None))
        native_df = self.snowpark.create_dataframe(rows, schema=columns)
        return native_df.write.saveAsTable(table_name, mode='append')  

