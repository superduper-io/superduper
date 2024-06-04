from superduperdb import superduper

user = 'default'
password = ''
port = 8123
host = 'localhost'

db = superduper(
    f"clickhouse://{user}:{password}@{host}:{port}", metadata_store=f'mongomock://meta'
)
