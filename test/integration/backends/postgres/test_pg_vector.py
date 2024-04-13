import random
import warnings
import tempfile
import ibis

import lorem
import psycopg2
import pytest

import superduperdb as s
from superduperdb import CFG, superduper
from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.base.datalayer import Datalayer
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.ibis.query import Table
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.model import ObjectModel
from superduperdb.components.vector_index import VectorIndex, vector
from superduperdb.components.schema import Schema
from superduperdb.backends.ibis.field_types import dtype


@pytest.fixture
def postgres_conn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = f'{tmp_dir}/mydb.sqlite'
        yield ibis.connect('postgres://' + str(tmp_db)), tmp_dir

@pytest.fixture
def test_db(postgres_conn):
    connection, tmp_dir = postgres_conn
    yield make_ibis_db(connection, connection, tmp_dir)


def make_ibis_db(db_conn, metadata_conn, tmp_dir, in_memory=False):
    return Datalayer(
        databackend=IbisDataBackend(conn=db_conn, name='ibis', in_memory=in_memory),
        metadata=SQLAlchemyMetadata(conn=metadata_conn.con, name='ibis'),
        artifact_store=FileSystemArtifactStore(conn=tmp_dir, name='ibis'),
    )


def random_vector_model(x):
    return [random.random() for _ in range(16)]


@pytest.fixture()
def pgvector_search_config():
    previous = s.CFG.vector_search
    s.CFG.vector_search = s.CFG.data_backend
    yield
    s.CFG.vector_search = previous


@pytest.mark.skipif(DO_SKIP, reason='Only pgvector deployments relevant.')
def test_setup_pgvector_vector_search(pgvector_search_config):
    model = ObjectModel(
        identifier='test-model', object=random_vector_model, encoder=vector(shape=(16,))
    )
    db = superduper()
    schema = Schema(
        identifier='docs-schema',
        fields={
            'text': dtype('str', schema=schema),
        },
    )
    table = Table('docs', schema=schema)

    vector_indexes = db.vector_indices  

    assert not vector_indexes

    db.execute(
        table.insert_many(
            [Document({'text': lorem.sentence()}) for _ in range(50)]
        )
    )
    db.add(
        VectorIndex(
            'test-vector-index',
            indexing_listener=Listener(
                model=model,
                key='text',
                select=table.select('text'),
            ),
        )
    )

    assert 'test-vector-index' in db.show('vector_index')
    assert 'test-vector-index' in db.vector_indices


@pytest.mark.skipif(DO_SKIP, reason='Only pgvector deployments relevant.')
def test_use_pgvector_vector_search(pgvector_search_config):
    db = superduper()
    schema = Schema(
        identifier='docs-schema',
        fields={
            'text': dtype('str', schema=schema),
        },
    )
    table = Table('docs', schema=schema)

    query = table.like(
        Document({'text': 'This is a test'}), n=5, vector_index='test-vector-index'
    ).find()

    it = 0
    for r in db.execute(query):
        print(r)
        it += 1

    assert it == 5
