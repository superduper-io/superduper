from superduperdb import superduper

user = 'superduper'
password = 'superduper'
port = 1521
host = 'localhost'

db = superduper(f"oracle://{user}:{password}@{host}:{port}")