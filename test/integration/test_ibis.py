import tempfile

import ibis
import PIL.Image
import pytest
import torch.nn
import torchvision

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import RawSQL, Table
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document as D
from superduperdb.components.schema import Schema
from superduperdb.ext.pillow.encoder import pil_image
from superduperdb.ext.torch.encoder import tensor
from superduperdb.ext.torch.model import TorchModel


@pytest.fixture
def sqllite_conn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = f'{tmp_dir}/mydb.sqlite'
        yield ibis.connect('sqlite://' + str(tmp_db)), tmp_dir


@pytest.fixture
def duckdb_conn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = f'{tmp_dir}/mydb.ddb'
        yield ibis.connect('duckdb://' + str(tmp_db)), tmp_dir


@pytest.fixture
def ibis_sqllite_db(sqllite_conn):
    connection, tmp_dir = sqllite_conn
    yield make_ibis_db(connection, connection, tmp_dir)


@pytest.fixture
def ibis_duckdb(duckdb_conn):
    connection, tmp_dir = duckdb_conn
    yield make_ibis_db(connection, connection, tmp_dir)


@pytest.fixture
def ibis_pandas_db(sqllite_conn):
    metadata_connection, tmp_dir = sqllite_conn
    connection = ibis.pandas.connect({})
    yield make_ibis_db(connection, metadata_connection, tmp_dir, in_memory=True)


def make_ibis_db(db_conn, metadata_conn, tmp_dir, in_memory=False):
    return Datalayer(
        databackend=IbisDataBackend(conn=db_conn, name='ibis', in_memory=in_memory),
        metadata=SQLAlchemyMetadata(conn=metadata_conn.con, name='ibis'),
        artifact_store=FileSystemArtifactStore(conn=tmp_dir, name='ibis'),
    )


def end2end_workflow(ibis_db, memory_table=False):
    db = ibis_db
    schema = Schema(
        identifier='my_table',
        fields={
            'id': dtype('str'),
            'health': dtype('int32'),
            'age': dtype('int32'),
            'image': pil_image,
        },
    )
    im = PIL.Image.open('test/material/data/test-image.jpeg')

    data_to_insert = [
        {'id': '1', 'health': 0, 'age': 25, 'image': im},
        {'id': '2', 'health': 1, 'age': 26, 'image': im},
        {'id': '3', 'health': 0, 'age': 27, 'image': im},
        {'id': '4', 'health': 1, 'age': 28, 'image': im},
    ]

    t = Table(identifier='my_table', schema=schema)

    db.add(t)

    insert = t.insert(
        [
            D(
                {
                    'id': d['id'],
                    'health': d['health'],
                    'age': d['age'],
                    'image': d['image'],
                }
            )
            for d in data_to_insert
        ]
    )
    db.execute(insert)

    q = t.select('image', 'age', 'health')

    result = db.execute(q)
    for img in result:
        img = img.unpack()
        assert isinstance(img['image'], PIL.Image.Image)
        assert isinstance(img['age'], int)
        assert isinstance(img['health'], int)

    # preprocessing function
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def postprocess(x):
        return int(x.topk(1)[1].item())

    # create a torchvision model
    resnet = TorchModel(
        identifier='resnet18',
        preprocess=preprocess,
        postprocess=postprocess,
        object=torchvision.models.resnet18(pretrained=False),
        datatype=dtype('int32'),
    )

    # Apply the torchvision model
    resnet.predict_in_db(
        X='image',
        db=db,
        select=t.select('id', 'image'),
        max_chunk_size=3000,
        overwrite=True,
    )

    # also add a vectorizing model
    vectorize = TorchModel(
        preprocess=lambda x: torch.randn(32),
        object=torch.nn.Linear(32, 16),
        identifier='model_linear_a',
        datatype=tensor(torch.float, (16,)),
    )

    # apply to the table
    vectorize.predict_in_db(
        X='image',
        db=db,
        select=t.select('id', 'image'),
        max_chunk_size=3000,
        overwrite=True,
    )

    # Build query to get the results back
    q = t.outputs(image='resnet18').select('id', 'image', 'age').filter(t.age > 25)

    # Get the results
    result = list(db.execute(q))
    assert result
    assert 'image' in result[0].unpack()

    # Get vector results
    q = (
        t.select('id', 'image', 'age')
        .filter(t.age > 25)
        .outputs(image='model_linear_a')
    )

    # Get the results
    result = list(db.execute(q))
    assert '_outputs.image.model_linear_a.0' in result[0].unpack()

    # Raw query
    if not memory_table:
        query = RawSQL(query='SELECT id from my_table')
        rows = list(db.execute(query))
        assert 'id' in list(rows[0].unpack().keys())
        assert [r.unpack() for r in rows] == [
            {'id': '1'},
            {'id': '2'},
            {'id': '3'},
            {'id': '4'},
        ]


def test_nested_query(ibis_sqllite_db):
    db = ibis_sqllite_db

    schema = Schema(
        identifier='my_table',
        fields={
            'id': dtype('int64'),
            'health': dtype('int32'),
            'age': dtype('int32'),
            'image': pil_image,
        },
    )

    t = Table(identifier='my_table', schema=schema)

    db.add(t)

    q = t.filter(t.age > 10)

    expr_, _ = q.compile(db)

    assert 'SELECT t0.id, t0.health, t0.age, t0.image, t0._fold' in str(expr_.compile())


def test_end2end_sql(ibis_sqllite_db):
    end2end_workflow(ibis_sqllite_db)


def test_end2end_duckdb(ibis_duckdb):
    end2end_workflow(ibis_duckdb)


def test_end2end_pandas(ibis_pandas_db):
    end2end_workflow(ibis_pandas_db, memory_table=True)
