from superduperdb import superduper

user = 'superduper'
password = 'superduper'
port = 5432
host = 'localhost'
database = 'test_db'
db_uri = f"postgres://{user}:{password}@{host}:{port}/{database}"

db = superduper(db_uri, metadata_store=db_uri.replace('postgres://', 'postgresql://')) 