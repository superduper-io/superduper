import os
import random
import shutil
import tempfile
import time
from contextlib import contextmanager

import ibis
import pytest
from fastapi.testclient import TestClient

try:
    import torch

    from superduperdb.ext.torch.model import TorchModel
except ImportError:
    torch = None

from superduperdb.backends.ibis.data_backend import IbisDataBackend
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.local.artifacts import FileSystemArtifactStore
from superduperdb.backends.mongodb.query import Collection
from superduperdb.backends.sqlalchemy.metadata import SQLAlchemyMetadata
from superduperdb.base.config import PollingStrategy
from superduperdb.base.datalayer import Datalayer
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.torch.encoder import tensor

RETRY_TIMEOUT = 1
LISTEN_TIMEOUT = 0.1


# NOTE 1:
# Some environments take longer than others for the changes to appear. For this
# reason this module has a special retry wrapper function.
#
# If you find yourself experiencing non-deterministic test runs which are linked
# to this module, consider increasing the number of retry attempts.
#
# TODO: this should add be done with a callback when the changes are ready.

# NOTE 2:
# Each fixture writes to a collection with a unique name. This means that the
# tests can be run in parallel without interactions between the tests. Be very
# careful if you find yourself changing the name of a collection in this module...

# NOTE 3:
# TODO: Modify this module so that the tests are actually run in parallel...


def make_ibis_db(db_conn, metadata_conn, tmp_dir, in_memory=False):
    return Datalayer(
        databackend=IbisDataBackend(conn=db_conn, name='ibis', in_memory=in_memory),
        metadata=SQLAlchemyMetadata(conn=metadata_conn.con, name='ibis'),
        artifact_store=FileSystemArtifactStore(conn=tmp_dir, name='ibis'),
    )


@pytest.fixture
def ibis_duckdb(duckdb_conn):
    uri, connection, tmp_dir = duckdb_conn
    yield uri, tmp_dir, make_ibis_db(connection, connection, tmp_dir)


@pytest.fixture
def duckdb_conn():
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_db = f'{tmp_dir}/mydb.ddb'
        uri = 'duckdb://' + str(tmp_db)
        yield uri, ibis.connect(uri), tmp_dir


def add_models_encoders(db, table):
    db.add(tensor(torch.float, shape=(32,)))
    db.add(tensor(torch.float, shape=(16,)))
    db.add(
        TorchModel(
            object=torch.nn.Linear(32, 16),
            identifier='model_linear_a',
            datatype='torch.float32[16]',
        )
    )
    db.add(
        Listener(
            select=table.select('id', 'x', 'y', 'z'),
            key='x',
            model='model_linear_a',
        )
    )
    db.add(
        Listener(
            select=table.select('id', 'x', 'y', 'z'),
            key='z',
            model='model_linear_a',
        )
    )
    vi = VectorIndex(
        identifier='test_index',
        indexing_listener='model_linear_a/x',
        compatible_listener='model_linear_a/z',
    )
    db.add(vi)
    return db


@pytest.fixture
def sql_database_with_cdc(ibis_duckdb, monkeypatch):
    import torch

    from superduperdb.backends.ibis.query import Table
    from superduperdb.components.schema import Schema
    from superduperdb.ext.torch.encoder import tensor

    datatype = tensor(torch.float, [32])

    schema = Schema(
        identifier='my_table',
        fields={
            'id': dtype('str'),
            'x': datatype,
            'y': dtype('int32'),
            'z': datatype,
            'auto_increment_field': dtype('int32'),
        },
    )

    table = Table('documents', schema=schema)
    uri, _, db = ibis_duckdb

    db.add(table)
    db = add_models_encoders(db, table)

    from superduperdb import CFG

    cfg_artifact_store = CFG.artifact_store
    cfg_metadata_store = CFG.metadata_store
    cfg_in_memory = CFG.cluster.vector_search

    CFG.force_set('cluster.vector_search', 'lance')
    db.fast_vector_searchers['test_index'] = db.initialize_vector_searcher(
        'test_index',
        searcher_type='lance',
    )
    cfg_databackend = CFG.data_backend

    CFG.force_set('data_backend', uri)
    CFG.force_set('artifact_store', None)

    db.cdc._cdc_existing_collections = []
    from functools import partial

    db.rebuild = partial(db.rebuild, cfg=CFG)
    strategy = PollingStrategy(
        type='incremental', auto_increment_field='auto_increment_field', frequency=1
    )

    db.cdc.listen(on=table, timeout=LISTEN_TIMEOUT, strategy=strategy)
    db.cdc.cdc_change_handler._QUEUE_BATCH_SIZE = 1

    yield table, db

    CFG.force_set('cluster.vector_search', cfg_in_memory)
    CFG.force_set('data_backend', cfg_databackend)
    CFG.force_set('artifact_store', cfg_artifact_store)
    CFG.force_set('metadata_store', cfg_metadata_store)

    db.cdc.stop()
    try:
        shutil.rmtree('.superduperdb/vector_indices')
    except FileNotFoundError:
        pass


@pytest.fixture
def database_with_cdc(database_with_default_encoders_and_model, monkeypatch):
    db = database_with_default_encoders_and_model

    from superduperdb import CFG

    cfg_in_memory = CFG.cluster.vector_search
    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())
    CFG.force_set('cluster.vector_search', 'lance')
    db.fast_vector_searchers['test_index'] = db.initialize_vector_searcher(
        'test_index',
        searcher_type='lance',
    )

    db.cdc._cdc_existing_collections = []
    listener = db.cdc.listen(on=Collection('documents'), timeout=LISTEN_TIMEOUT)
    db.cdc.cdc_change_handler._QUEUE_BATCH_SIZE = 1

    yield listener, 'documents', db

    CFG.force_set('cluster.vector_search', cfg_in_memory)
    db.cdc.stop()
    try:
        shutil.rmtree('.superduperdb/vector_indices')
    except FileNotFoundError:
        pass


def retry_state_check(state_check, retry_timeout=None, sleep=0.1):
    start = time.time()

    exc_msg = ''
    retry_timeout = retry_timeout if retry_timeout else RETRY_TIMEOUT
    while (time.time() - start) < retry_timeout:
        try:
            return state_check()
        except Exception as e:
            exc_msg = str(e)
            time.sleep(sleep)

    raise Exception(exc_msg)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_smoke(database_with_cdc):
    """Health-check before we test stateful database changes"""
    _, name, db = database_with_cdc
    assert isinstance(name, str)


@pytest.mark.parametrize('op_type', ['insert'])
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_task_workflow(
    database_with_cdc,
    fake_inserts,
    fake_updates,
    op_type,
):
    """Test that task graph executed on `insert`"""

    _, name, db = database_with_cdc

    with add_and_cleanup_listeners(
        db, Collection(name).find()
    ) as database_with_listeners:
        # `refresh=False` to ensure `_outputs` not produced after `Insert` refresh.
        data = None
        if op_type == 'insert':
            data = fake_inserts
        elif op_type == 'update':
            data = fake_updates

        inserted_ids, _ = database_with_listeners.execute(
            Collection(name).insert_many([data[0]]),
            refresh=False,
        )

        def state_check():
            doc = database_with_listeners.databackend.get_table_or_collection(
                name
            ).find_one({'_id': inserted_ids[0]})
            assert '_outputs' in list(doc.keys())

        retry_state_check(state_check)

        # state_check_2 can't be merged with state_check because the
        # '_outputs' key needs to be present in 'doc'
        def state_check_2():
            doc = database_with_listeners.databackend.get_table_or_collection(
                name
            ).find_one({'_id': inserted_ids[0]})
            state = []
            state.append('model_linear_a' in doc['_outputs']['x'].keys())
            state.append('model_linear_a' in doc['_outputs']['z'].keys())
            assert all(state)

        retry_state_check(state_check_2)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_vector_database_sync_with_delete(
    database_with_cdc,
    fake_inserts,
):
    (
        _,
        name,
        db,
    ) = database_with_cdc

    inserted_ids, _ = db.execute(
        Collection(name).insert_many(fake_inserts[:2]),
        refresh=False,
    )

    def state_check():
        assert len(db.fast_vector_searchers['test_index']) == 2

    retry_state_check(state_check)

    # TODO DatabaseListener sees the change but CDCHandler doesn't
    db.execute(
        Collection(name).delete_one({'_id': inserted_ids[0]}),
        refresh=False,
    )

    # check if vector database is in sync with the model outputs
    def state_check_2():
        assert len(db.fast_vector_searchers['test_index']) == 1

    retry_state_check(state_check_2)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_vector_database_sync(
    database_with_cdc,
    fake_inserts,
):
    _, name, db = database_with_cdc
    db.execute(
        Collection(name).insert_many([fake_inserts[0]]),
        refresh=False,
    )

    # Check if vector database is in sync with the model outputs
    def state_check():
        vector_searcher = db.fast_vector_searchers['test_index']
        assert len(vector_searcher) == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_single_insert(
    database_with_cdc,
    fake_inserts,
):
    listener, name, db = database_with_cdc
    db.execute(
        Collection(name).insert_many([fake_inserts[0]]),
        refresh=False,
    )

    def state_check():
        assert listener.info()["inserts"] == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_many_insert(
    database_with_cdc,
    fake_inserts,
):
    listener, name, db = database_with_cdc
    db.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )

    def state_check():
        assert listener.info()["inserts"] == len(fake_inserts)

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_delete_one(
    database_with_cdc,
    fake_inserts,
):
    listener, name, db = database_with_cdc
    db.cdc.stop()
    inserted_ids, _ = db.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )
    listener = db.cdc.listen(on=Collection('documents'), timeout=LISTEN_TIMEOUT)

    db.execute(Collection(name).delete_one({'_id': inserted_ids[0]}), refresh=False)

    def state_check():
        assert listener.info()["deletes"] == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_many_update(
    database_with_cdc,
    fake_updates,
):
    listener, name, db = database_with_cdc
    db.cdc.stop()
    inserted_ids, _ = db.execute(
        Collection(name).insert_many(fake_updates), refresh=False
    )
    encoder = db.datatypes['torch.float32[32]']
    listener = db.cdc.listen(on=Collection('documents'), timeout=LISTEN_TIMEOUT)

    db.execute(
        Collection(name).update_many(
            {"_id": {"$in": inserted_ids[:5]}},
            Document({'$set': {'x': encoder(torch.randn(32))}}),
        ),
        refresh=False,
    )

    def state_check():
        assert listener.info()["updates"] == 5

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_without_cdc_handler(
    database_with_cdc,
    fake_inserts,
):
    """Test that `insert` without CDC handler does not execute task graph"""
    _, name, db = database_with_cdc
    inserted_ids, _ = db.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )
    doc = db.execute(Collection(name).find_one({'_id': inserted_ids[0]}))
    assert '_outputs' not in doc.keys()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_cdc_stop(database_with_cdc):
    """Test that CDC listen service stopped properly"""
    listener, _, _ = database_with_cdc
    listener.stop()

    def state_check():
        assert not listener._scheduler.is_alive()

    retry_state_check(state_check)


@contextmanager
def add_and_cleanup_listeners(database, select):
    """Add listeners to the database and remove them after the test"""

    m = database.load('model', 'model_linear_a')

    listener_x = Listener(
        key='x',
        model=m,
        select=select,
    )

    listener_z = Listener(
        key='z',
        model=m,
        select=select,
    )

    database.add(listener_x)
    database.add(listener_z)

    yield database


@pytest.mark.skip(reason='Duckdb is not threadsafe')
@pytest.mark.parametrize('op_type', ['insert'])
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_sql_task_workflow(
    sql_database_with_cdc,
    op_type,
):
    """Test that task graph executed on `insert`"""

    table, db = sql_database_with_cdc

    # `refresh=False` to ensure `_outputs` not produced after `Insert` refresh.
    data = []
    for i in range(10):
        x = torch.randn(32)
        y = int(random.random() > 0.5)
        z = torch.randn(32)
        data.append(
            Document({'id': str(i), 'x': x, 'y': y, 'z': z, 'auto_increment_field': i})
        )
    _ = db.execute(
        table.insert([data[0]]),
        refresh=False,
    )
    time.sleep(4)
    db.cdc.stop()

    def state_check():
        t = db.databackend.conn.table('_outputs_model_linear_a_0')
        outputs = t.select('_input_id').execute()
        print(outputs)
        assert len(outputs) == 1

    retry_state_check(state_check, 3, 1)


@pytest.fixture
def client(monkeypatch, database_with_default_encoders_and_model):
    from superduperdb import CFG

    monkeypatch.setattr(
        CFG.cluster.cdc, 'uri', os.environ['SUPERDUPERDB_CLUSTER_CDC_URI']
    )
    database_with_default_encoders_and_model.cdc.start()
    from superduperdb.cdc.app import app as cdc_app

    cdc_app.app.state.pool = database_with_default_encoders_and_model
    client = TestClient(cdc_app.app)
    yield client

    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())


def test_basic_workflow(client):
    listener = 'model_linear_a/x'
    response = client.get(f"/listener/add?name={listener}")
    assert response.status_code == 200

    db = client.app.state.pool
    assert 'documents' in db.cdc._CDC_LISTENERS

    response = client.get(f"/listener/delete?name={listener}")
    assert 'documents' not in db.cdc._CDC_LISTENERS
