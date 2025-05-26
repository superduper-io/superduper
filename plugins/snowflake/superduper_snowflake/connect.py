import os
import re
import threading

from snowflake.snowpark import Session
from superduper import logging
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

db_lock = threading.Lock()
SESSION_DIR = os.environ.get('SNOWFLAKE_SESSION_DIR') or '/snowflake/session'


class _SnowflakeTokenWatcher(FileSystemEventHandler):
    timeout = 60

    def __init__(self, databackend):
        super().__init__()
        self.databackend = databackend

    def on_any_event(self, event):
        logging.warn(str(event))

        if event.src_path.endswith('data_tmp') and event.event_type == 'moved':
            with db_lock:
                logging.info(
                    f'Detected Snowflake token file change, '
                    f'reconnect to the {self.databackend.__class__.__name__}'
                )
                self.databackend.reconnect()


def watch_token_file(databackend):
    """Watch the Snowflake token file for changes.

    :param databackend: The data backend instance to reconnect.
    This function sets up a file system observer that watches The
    Snowflake token file for changes. When the token file is modified,
    it will trigger a reconnection of the data backend.
    """
    observer = Observer()
    handler = _SnowflakeTokenWatcher(databackend)

    logging.info(f'Starting Snowflake token watcher on {SESSION_DIR}/token')

    observer.schedule(handler, SESSION_DIR, recursive=False)
    observer.start()
    logging.info('Started Snowflake token watcher')
    return observer


def connect(uri):
    """Connect to Snowflake using the Snowpark session.

    :param uri: The URI of the Snowflake connection.

    If the URI is 'snowflake://', the connection parameters are read
    from environment variables.

    - SNOWFLAKE_HOST: The Snowflake host.
    - SNOWFLAKE_PORT: The Snowflake port.
    - SNOWFLAKE_ACCOUNT: The Snowflake account.
    - SNOWFLAKE_WAREHOUSE: The Snowflake warehouse.
    - SNOWFLAKE_DATABASE: The Snowflake database.
    - SUPERDUPER_DATA_SCHEMA: The Snowflake schema.
    - /snowflake/session/token: The Snowflake OAuth token.
    """
    logging.info(
        'Creating Snowpark session for' ' snowflake data-backend implementation'
    )
    if uri == 'snowflake://':
        try:
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
            schema = os.environ['SUPERDUPER_DATA_SCHEMA']
        except KeyError as e:
            all_snowflake_envs = [
                key for key in os.environ.keys() if 'SNOWFLAKE' in key
            ]
            raise KeyError(
                f'Environment variable {e} not set. '
                f'Available snowflake environment variables: {all_snowflake_envs}. '
                'Please set the environment variables for Snowflake connection.'
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

    return Session.builder.configs(connection_parameters).create(), schema
