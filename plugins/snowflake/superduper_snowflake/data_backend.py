from functools import wraps
import os
import re
import threading
import time

import ibis
import pandas
import snowflake.connector

from superduper_snowflake.schema import (
    ibis_schema_to_snowpark_cols,
    snowpark_cols_to_schema,
)
from superduper import logging
from superduper_ibis.data_backend import IbisDataBackend
from snowflake.snowpark import Session

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer


db_lock = threading.Lock()
SESSION_DIR = '/snowflake/session'


# Create token watcher with watchdog on /session/token
class SnowflakeTokenWatcher(PatternMatchingEventHandler):
    timeout = 60
    def __init__(self, databackend):
        # Only match the exact file
        super().__init__(patterns=['token'], ignore_directories=True)
        self.databackend = databackend

    def on_deleted(self, event):
        logging.warn(f'{event.src_path} has been deleted')
    
    def on_modified(self, event):
        logging.info(f"{event.src_path} has been modified")
        with db_lock:
            self.databackend.reconnect()

    def on_created(self, event):
        logging.info(f"{event.src_path} has been created")
        with db_lock:
            self.databackend.reconnect()
            self.databackend.datalayer.metadata.reconnect()


def watch_token_file(databackend):
    observer = Observer()
    handler = SnowflakeTokenWatcher(databackend)

    observer.schedule(handler, SESSION_DIR, recursive=False)
    observer.start()
    logging.info('Started Snowflake token watcher')
    return observer


class SnowflakeDataBackend(IbisDataBackend):
    @wraps(IbisDataBackend.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snowpark = self._get_snowpark_session(self.uri)
        self.observer = None
    # if self.uri == 'snowflake://':
        self.observer = watch_token_file(self)

    def disconnect(self):
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
        super().disconnect()
        self.snowpark.close()

    @staticmethod
    def _get_snowpark_session(uri):
        logging.info(
            'Creating Snowpark session for' ' snowflake vector-search implementation'
        )
        if uri == 'snowflake://':
            connection_parameters = dict(
                host=os.environ['SNOWFLAKE_HOST'],
                port=int(os.environ['SNOWFLAKE_PORT']),
                account=os.environ['SNOWFLAKE_ACCOUNT'],
                authenticator='oauth',
                token=open(SESSION_DIR + '/token').read(),
                warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
                database=os.environ['SNOWFLAKE_DATABASE'],
                schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
            )
        else:
            if '?warehouse=' not in uri:
                match = re.match('^snowflake:\/\/(.*):(.*)\@(.*)\/(.*)\/(.*)$', uri)
                user, password, account, database, schema = match.groups()
                warehouse = None
            else:
                match = re.match(
                    '^snowflake://(.*):(.*)@(.*)/(.*)/(.*)?warehouse=(.*)$', uri
                )
                user, password, account, database, schema, warehouse = match.groups()

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
            token=open(SESSION_DIR + '/token').read(),
            warehouse=os.environ['SNOWFLAKE_WAREHOUSE'],
            database=os.environ['SNOWFLAKE_DATABASE'],
            schema=os.environ['SUPERDUPER_DATA_SCHEMA'],
        )

    def _connection_callback(self, uri):
        if uri != 'snowflake://':
            return IbisDataBackend._connection_callback(uri)
        return (
            ibis.snowflake.from_connection(
                self._do_connection_callback(uri), create_object_udfs=False
            ),
            'snowflake',
            False,
        )

    def reconnect(self):
        logging.info('Reconnecting to Snowflake')
        super().reconnect()
        self.snowpark = self._get_snowpark_session(self.uri)
        logging.info('Reconnecting to Snowflake... DONE')

    def insert(self, table_name, raw_documents):
        ibis_schema = self.conn.table(table_name).schema()
        df = pandas.DataFrame(raw_documents)
        rows = list(df.itertuples(index=False, name=None))
        columns = list(ibis_schema.keys())
        df = df.to_dict(orient='records')
        get_row = lambda row: [row[col] for col in columns]
        rows = list(map(get_row, df))
        snowpark_cols = ibis_schema_to_snowpark_cols(ibis_schema)
        snowpark_schema = snowpark_cols_to_schema(snowpark_cols, columns)
        native_df = self.snowpark.create_dataframe(rows, schema=snowpark_schema)
        return native_df.write.saveAsTable(f'"{table_name}"', mode='append')

    def raw_sql(self, query):
        return self.conn.raw_sql(query)
