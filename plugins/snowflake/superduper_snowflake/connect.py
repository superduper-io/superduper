import os
import re

from snowflake.snowpark import Session

from superduper import logging


def connect(uri):
    logging.info(
        'Creating Snowpark session for'
        ' snowflake data-backend implementation'
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