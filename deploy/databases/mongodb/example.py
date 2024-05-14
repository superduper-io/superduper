from superduperdb import superduper

user = 'superduper'
password = 'superduper'
port = 27017
host = 'localhost'
database = 'test_db'

db = superduper(f"mongodb://{user}:{password}@{host}:{port}/{database}")