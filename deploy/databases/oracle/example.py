import ibis

user = 'superduper'
password = 'superduper'
port = 1521
host = 'localhost'
database = 'test_db'

# OK
con = ibis.oracle.connect(
    user='superduper',
    password='superduper',
    port=1521,
    host='localhost',
    database='test_db',
)

# ERROR AttributeError: 'str' object has no attribute 'username'
# con = ibis.connect(f"oracle://{user}:{password}@{host}:{port}/{database}")

# ERROR ModuleNotFoundError: No module named 'ibis.backends.base'
# from superduperdb import superduper
# db = superduper(f"oracle://{user}:{password}@{host}:{port}/{database}")
