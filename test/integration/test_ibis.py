import ibis
import PIL.Image
import pandas
from sklearn import svm
#import torchvision

from superduperdb import superduper
from superduperdb import CFG
from superduperdb.container.encoder import Encoder
from superduperdb.container.schema import Schema
from superduperdb.db.base.build import build_vector_database
from superduperdb.container.document import Document as D
from superduperdb.db.ibis.db import IbisDB
from superduperdb.db.filesystem.artifacts import FileSystemArtifactStore
from superduperdb.db.ibis.data_backend import IbisDataBackend
from superduperdb.db.ibis.schema import IbisSchema
from superduperdb.db.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.db.ibis.query import Table
from superduperdb.ext.pillow.image import pil_image
from superduperdb.ext.torch.model import TorchModel
# from superduperdb.container.schema import Schema


import os
os.remove('mydb.sqlite')
connection = ibis.sqlite.connect("mydb.sqlite")

# Insert some sample data into the table
schema = ibis.schema({'id': 'int64', 'health': 'int32', 'age': 'int32'})
# Create the table
table_name = 'my_table'

# connection.create_table(table_name, schema=schema)


# create data layer
db = IbisDB(
    databackend=IbisDataBackend(conn=connection, name='ibis'),
    metadata=SQLAlchemyMetadata(conn=connection.con, name='ibis'),
    artifact_store=FileSystemArtifactStore(
        conn='./.tmp', name='ibis'
    ),
    vector_database=build_vector_database(CFG.vector_search.type),
)

schema = IbisSchema(
    identifier='my_table',
    fields={
        'id': 'int64',
        'health': 'int32',
        'age': 'int32',
        'image': pil_image,
    }
)
im = PIL.Image.open('test/material/data/test-image.jpeg')
data_to_insert = [
    {'id': 1, 'health': 0, 'age': 25, 'image': im},
    {'id': 2, 'health': 0, 'age': 25, 'image': im},
    {'id': 3, 'health': 0, 'age': 25, 'image': im},
    {'id': 4, 'health': 0, 'age': 25, 'image': im},
]
t = Table(identifier='my_table', schema=schema)
t.create(db)
im_b = open('test/material/data/test-image.jpeg', 'rb').read()
db.add(t)
db.execute(
    t.insert([D({'id': d['id'], 'health': d['health'], 'age': d['age'], 'image': d['image']}) for d in data_to_insert])
)



# -------------- retrieve data  from table ----------------
imgs = db.execute(t.select("image", "age", "health"))
for img in imgs:
    print(img)


import torchvision


# preprocessing function
preprocess = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)])

def postprocess(x):
    return int(x.topk(1)[1].item())

# create a torchvision model
resnet = TorchModel(
        identifier='resnet18',
    preprocess=preprocess,
    postprocess=postprocess,

    object=torchvision.models.resnet18(pretrained=False),
)

# apply the torchvision model
resnet.predict(X='image', db=db, select=t.select('id', 'image'), max_chunk_size=3000, overwrite=True)



# Query the results back
q = t.filter(t.age == 25).outputs('resnet18', db)
curr = db.execute(q)
for c in curr:
    print(c)




breakpoint()



# ------------------ SKLEARN ------------------
# add a sklearn mode
svc = svm.SVC(gamma='scale', class_weight='balanced', C=100, verbose=True)
svc.fit(pandas.DataFrame(data_to_insert).iloc[:, :3], [0, 1, 1, 1, 0])

model = superduper(
    svc,
    postprocess=lambda x: int(x),
    preprocess=lambda x: list(x.values()
))

model.predict(X='_base', db=db, select=t.filter(t.age > 25), max_chunk_size=3000, overwrite=True)    

#  Query result back
out_table = connection.table(model.identifier)
q = t.filter(t.age > 26).outputs(model.identifier, db)

curr = db.execute(q)
for c in curr:
    print(c)
