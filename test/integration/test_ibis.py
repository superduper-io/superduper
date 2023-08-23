import ibis
import sqlalchemy
from sklearn import svm

from superduperdb import superduper
from superduperdb import CFG
from superduperdb.db.base.build import build_vector_database
from superduperdb.container.document import Document as D
from superduperdb.db.ibis.db import IbisDB
from superduperdb.db.filesystem.artifacts import FileSystemArtifactStore
from superduperdb.db.ibis.data_backend import IbisDataBackend
from superduperdb.db.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.db.ibis.query import Table






connection_sql=sqlalchemy.create_engine('sqlite:///./mydb.sqlite')
connection = ibis.sqlite.connect("mydb.sqlite")



# Insert some sample data into the table
schema = ibis.schema({'id': 'int64', 'health': 'int32', 'age': 'int32'})
# Create the table
table_name = 'my_table'
connection.create_table(table_name, schema=schema)

data_to_insert = [
    (1, 0, 25),
    (2, 1, 26),
    (3, 1, 27),
    (4, 1, 28),
    (5, 0, 29),
]
connection.insert(table_name, data_to_insert)




# create data layer
db = IbisDB(
    databackend=IbisDataBackend(conn=connection, name='ibis'),
    metadata=SQLAlchemyMetadata(conn=connection_sql, name='ibis'),
    artifact_store=FileSystemArtifactStore(
        conn='./.tmp', name='ibis'
    ),
    vector_database=build_vector_database(CFG.vector_search.type),
)


# add a sklearn model
table = Table(name='my_table')
svc = svm.SVC(gamma='scale', class_weight='balanced', C=100, verbose=True)
svc.fit(data_to_insert, [0, 1, 1, 1, 0])

model = superduper(
        svc,
    postprocess=lambda x: int(x),
    preprocess=lambda x: list(x.values()
))
db.add(model)



# Do prediction
m = db.load('model', 'svc')
m.predict(X='_base', db=db, select=table.filter(table.age > 25), max_chunk_size=3000, overwrite=True)    




#  Query result back
out_table = connection.table(m.identifier)
q = table.filter(table.age > 26).outputs(m.identifier, db)

curr = db.execute(q)

for c in curr:
    print(c)
