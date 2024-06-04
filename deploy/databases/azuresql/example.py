from superduperdb import superduper

user = 'sa'
password = 'Superduper#1'
port = 1433
host = 'localhost'

db = superduper(f"mssql://{user}:{password}@{host}:{port}")
