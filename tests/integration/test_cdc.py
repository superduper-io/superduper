import pytest
import time
from unittest.mock import MagicMock
from unittest.mock import patch

from superduperdb.datalayer.base.cdc import DatabaseWatcher
from superduperdb.queries.mongodb.queries import Collection
from superduperdb.misc.task_queue import cdc_queue


@patch('superduperdb.datalayer.mongodb.cdc.copy_vectors')
@pytest.fixture()
def watch_fixture(with_vector_index):
    watcher = DatabaseWatcher(db=with_vector_index, on=Collection(name='documents'))
    yield watcher
    watcher.stop()


class TestMongoCDC:
    def teardown_method(self):
        with cdc_queue.mutex:
            cdc_queue.queue.clear()

    def test_smoke(self, watch_fixture):
        watch_fixture.watch()
        watch_fixture.stop()

    def test_single_insert(self, watch_fixture, with_vector_index, an_single_insert):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        with_vector_index.execute(
            Collection(name='documents').insert_many([an_single_insert])
        )
        time.sleep(3)
        info = watch_fixture.info()
        watch_fixture.stop()
        assert info['inserts'] == 1

    def test_many_insert(self, watch_fixture, with_vector_index, an_insert):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        with_vector_index.execute(Collection(name='documents').insert_many(an_insert))
        time.sleep(3)
        info = watch_fixture.info()
        watch_fixture.stop()
        assert info['inserts'] == len(an_insert)

    @patch('superduperdb.datalayer.mongodb.cdc.copy_vectors')
    def test_task_workflow_on_insert(
        self, mocked_copy_vectors, watch_fixture, with_vector_index, an_single_insert
    ):
        watch_fixture.watch()

        # Adding this so that we ensure the _outputs where not produce
        # after Insert query refresh.
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many([an_single_insert], refresh=False)
        )
        time.sleep(3)
        _id = output.inserted_ids[0]
        time.sleep(8)
        doc = with_vector_index.db['documents'].find_one({'_id': _id})
        watch_fixture.stop()
        assert '_outputs' in doc.keys()
        assert 'linear_a' in doc['_outputs']['x'].keys()
        assert 'linear_a' in doc['_outputs']['z'].keys()

    def test_cdc_stop(self, watch_fixture):
        watch_fixture.watch()

        _prev_states = [
            watch_fixture._scheduler.is_alive(),
            watch_fixture._cdc_change_handler.is_alive(),
        ]

        watch_fixture.stop()
        time.sleep(4)
        _post_stop_states = [
            watch_fixture._scheduler.is_alive(),
            watch_fixture._cdc_change_handler.is_alive(),
        ]
        watch_fixture.stop()
        assert all(
            [
                _prev_s != _post_s
                for _prev_s, _post_s in zip(_prev_states, _post_stop_states)
            ]
        )

    def test_insert_with_cdc(self, watch_fixture, with_vector_index, an_insert):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many(an_insert, refresh=True)
        )

        _id = output.inserted_ids[0]
        doc = with_vector_index.db['documents'].find_one({'_id': _id})
        watch_fixture.stop()
        assert '_outputs' not in doc.keys()
