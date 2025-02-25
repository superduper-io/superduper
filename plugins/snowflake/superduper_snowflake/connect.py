import os
import re

from snowflake.snowpark import Session
from superduper import logging


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
