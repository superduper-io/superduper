import time
import uuid
from contextlib import contextmanager

import pytest
import tdir

try:
    import torch
except ImportError:
    torch = None

from superduperdb.container.document import Document
from superduperdb.container.listener import Listener
from superduperdb.db.mongodb.query import Collection
from superduperdb.vector_search.base import VectorCollectionConfig

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


@pytest.fixture
def listener_and_collection_name(database_with_default_encoders_and_model):
    db = database_with_default_encoders_and_model
    collection_name = str(uuid.uuid4())
    db.cdc._cdc_existing_collections = []
    listener = db.cdc.listen(on=Collection(collection_name), timeout=LISTEN_TIMEOUT)
    db.cdc.cdc_change_handler._QUEUE_BATCH_SIZE = 1

    yield listener, collection_name

    db.cdc.stop()


@pytest.fixture
def listener_with_vector_database(database_with_default_encoders_and_model):
    collection_name = str(uuid.uuid4())
    db = database_with_default_encoders_and_model
    with tdir():
        db.cdc._cdc_existing_collections = []
        listener = db.cdc.listen(on=Collection(collection_name), timeout=LISTEN_TIMEOUT)
        vector_db_client = db.vector_database

        yield listener, vector_db_client, collection_name

        db.cdc.stop()


@pytest.fixture
def listener_without_cdc_handler_and_collection_name(
    database_with_default_encoders_and_model,
):
    db = database_with_default_encoders_and_model
    collection_name = str(uuid.uuid4())
    db.cdc._cdc_existing_collections = []
    listener = db.cdc.listen(on=Collection(collection_name), timeout=LISTEN_TIMEOUT)
    yield listener, collection_name
    db.cdc.stop()


def retry_state_check(state_check):
    start = time.time()

    exc = None
    while (time.time() - start) < RETRY_TIMEOUT:
        try:
            return state_check()
        except Exception as e:
            exc = e
            time.sleep(0.1)

    raise Exception(exc)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_smoke(listener_without_cdc_handler_and_collection_name):
    """Health-check before we test stateful database changes"""
    _, name = listener_without_cdc_handler_and_collection_name
    assert isinstance(name, str)


@pytest.mark.parametrize('op_type', ['insert'])
@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_task_workflow(
    listener_and_collection_name,
    database_with_default_encoders_and_model,
    fake_inserts,
    fake_updates,
    op_type,
):
    """Test that task graph executed on `insert`"""

    listener, name = listener_and_collection_name

    with add_and_cleanup_listeners(
        database_with_default_encoders_and_model, name
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
    listener_with_vector_database,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    _, vector_db_client, name = listener_with_vector_database

    with add_and_cleanup_listeners(
        database_with_default_encoders_and_model, name
    ) as database_with_listeners:
        inserted_ids, _ = database_with_listeners.execute(
            Collection(name).insert_many([fake_inserts[0]]),
            refresh=False,
        )

        def state_check():
            table = vector_db_client.get_table(
                VectorCollectionConfig(id='model_linear_a/x', dimensions=0)
            )
            assert table.size() == 1

        retry_state_check(state_check)
        database_with_listeners.execute(
            Collection(name).delete_one({'_id': inserted_ids[0]})
        )

        # check if vector database is in sync with the model outputs
        def state_check_2():
            table = vector_db_client.get_table(
                VectorCollectionConfig(id='model_linear_a/x', dimensions=0)
            )
            assert table.size() == 0

        retry_state_check(state_check_2)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_vector_database_sync(
    listener_with_vector_database,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    listener, vector_db_client, name = listener_with_vector_database

    with add_and_cleanup_listeners(
        database_with_default_encoders_and_model, name
    ) as database_with_listeners:
        database_with_listeners.execute(
            Collection(name).insert_many([fake_inserts[0]]),
            refresh=False,
        )

        # Check if vector database is in sync with the model outputs
        def state_check():
            table = vector_db_client.get_table(
                VectorCollectionConfig(id='model_linear_a/x', dimensions=0)
            )
            assert table.size() == 1

        retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_single_insert(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    listener, name = listener_without_cdc_handler_and_collection_name
    database_with_default_encoders_and_model.execute(
        Collection(name).insert_many([fake_inserts[0]]),
        refresh=False,
    )

    def state_check():
        assert listener.info()["inserts"] == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_many_insert(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    listener, name = listener_without_cdc_handler_and_collection_name
    database_with_default_encoders_and_model.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )

    def state_check():
        assert listener.info()["inserts"] == len(fake_inserts)

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_delete_one(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    listener, name = listener_without_cdc_handler_and_collection_name
    inserted_ids, _ = database_with_default_encoders_and_model.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )

    database_with_default_encoders_and_model.execute(
        Collection(name).delete_one({'_id': inserted_ids[0]})
    )

    def state_check():
        assert listener.info()["deletes"] == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_single_update(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_updates,
):
    listener, name = listener_without_cdc_handler_and_collection_name
    inserted_ids, _ = database_with_default_encoders_and_model.execute(
        Collection(name).insert_many(fake_updates),
        refresh=False,
    )
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    database_with_default_encoders_and_model.execute(
        Collection(name).update_many(
            {"_id": inserted_ids[0]},
            Document({'$set': {'x': encoder(torch.randn(32))}}),
        )
    )

    def state_check():
        assert listener.info()["updates"] == 1

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_many_update(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_updates,
):
    listener, name = listener_without_cdc_handler_and_collection_name
    inserted_ids, _ = database_with_default_encoders_and_model.execute(
        Collection(name).insert_many(fake_updates), refresh=False
    )
    encoder = database_with_default_encoders_and_model.encoders['torch.float32[32]']
    database_with_default_encoders_and_model.execute(
        Collection(name).update_many(
            {"_id": {"$in": inserted_ids[:5]}},
            Document({'$set': {'x': encoder(torch.randn(32))}}),
        )
    )

    def state_check():
        assert listener.info()["updates"] == 5

    retry_state_check(state_check)


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_insert_without_cdc_handler(
    listener_without_cdc_handler_and_collection_name,
    database_with_default_encoders_and_model,
    fake_inserts,
):
    """Test that `insert` without CDC handler does not execute task graph"""
    _, name = listener_without_cdc_handler_and_collection_name
    inserted_ids, _ = database_with_default_encoders_and_model.execute(
        Collection(name).insert_many(fake_inserts),
        refresh=False,
    )
    db = database_with_default_encoders_and_model
    doc = db.execute(Collection(name).find_one({'_id': inserted_ids[0]}))
    assert '_outputs' not in doc.content.keys()


@pytest.mark.skipif(not torch, reason='Torch not installed')
def test_cdc_stop(listener_and_collection_name):
    """Test that CDC listen service stopped properly"""
    listener, _ = listener_and_collection_name
    listener.stop()

    def state_check():
        assert not listener._scheduler.is_alive()

    retry_state_check(state_check)


@contextmanager
def add_and_cleanup_listeners(database, collection_name):
    """Add listeners to the database and remove them after the test"""
    listener_x = Listener(
        key='x',
        model='model_linear_a',
        select=Collection(collection_name).find(),
    )

    listener_z = Listener(
        key='z',
        model='model_linear_a',
        select=Collection(collection_name).find(),
    )

    database.add(listener_x)
    database.add(listener_z)
    try:
        yield database
    finally:
        database.remove('listener', 'model_linear_a/x', force=True)
        database.remove('listener', 'model_linear_a/z', force=True)
