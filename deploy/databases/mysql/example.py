from superduper import superduper

user = 'superduper'
password = 'superduper'
port = 3306
host = 'localhost'
database = 'test_db'

db = superduper(f"mysql://{user}:{password}@{host}:{port}/{database}")