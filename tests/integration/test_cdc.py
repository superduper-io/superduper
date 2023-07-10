import pytest
import time
from unittest.mock import MagicMock
from unittest.mock import patch

import torch

from superduperdb.datalayer.base.cdc import DatabaseWatcher
from superduperdb.datalayer.mongodb.query import Collection
from superduperdb.misc.task_queue import cdc_queue


@patch('superduperdb.datalayer.mongodb.cdc.copy_vectors')
@pytest.fixture()
def watch_fixture(with_vector_index):
    watcher = DatabaseWatcher(db=with_vector_index, on=Collection(name='documents'))
    watcher._cdc_change_handler._QUEUE_TIMEOUT = 0
    watcher._cdc_change_handler._QUEUE_BATCH_SIZE = 1
    yield watcher
    watcher.stop()


class TestMongoCDC:
    def teardown_method(self):
        with cdc_queue.mutex:
            cdc_queue.queue.clear()

    def test_smoke(self, watch_fixture):
        watch_fixture.watch()
        watch_fixture.stop()

    def test_single_insert(self, watch_fixture, with_vector_index, a_single_insert):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        with_vector_index.execute(
            Collection(name='documents').insert_many([a_single_insert])
        )
        time.sleep(2)
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

    def test_single_update(
        self, watch_fixture, random_data, with_vector_index, an_insert
    ):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many(an_insert)
        )
        to_update = torch.randn(32)
        t = random_data.encoders['torch.float32[32]']
        with_vector_index.execute(
            Collection(name='documents').update_many(
                {"_id": output.inserted_ids[0]}, {'$set': {'x': t(to_update)}}
            )
        )
        time.sleep(2)
        info = watch_fixture.info()
        watch_fixture.stop()
        assert info['updates'] == 1

    def test_many_update(
        self, watch_fixture, random_data, with_vector_index, an_insert
    ):
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many(an_insert)
        )
        to_update = torch.randn(32)
        count = 5
        t = random_data.encoders['torch.float32[32]']
        find_query = {"_id": {"$in": output.inserted_ids[:count]}}
        with_vector_index.execute(
            Collection(name='documents').update_many(
                find_query, {'$set': {'x': t(to_update)}}
            )
        )

        time.sleep(2)
        info = watch_fixture.info()
        watch_fixture.stop()
        assert info['updates'] == count

    @pytest.mark.skip('Broken')
    @patch('superduperdb.datalayer.mongodb.cdc.copy_vectors')
    def test_task_workflow_on_insert(
        self, mocked_copy_vectors, watch_fixture, with_vector_index, a_single_insert
    ):
        """
        A test which checks if task graph executed on insert.
        task graph.
        """

        watch_fixture.watch()

        # Adding this so that we ensure the _outputs where not produce
        # after Insert query refresh.
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many([a_single_insert], refresh=False)
        )
        time.sleep(4)
        _id = output.inserted_ids[0]
        doc = with_vector_index.db['documents'].find_one({'_id': _id})
        import pdb

        pdb.set_trace()
        watch_fixture.stop()
        assert '_outputs' in doc.keys()
        assert 'linear_a' in doc['_outputs']['x'].keys()
        assert 'linear_a' in doc['_outputs']['z'].keys()

    def test_cdc_stop(self, watch_fixture):
        """
        A small test which tests if cdc watch service has stopped
        properly.
        """
        watch_fixture.watch()

        _prev_states = [
            watch_fixture._scheduler.is_alive(),
            watch_fixture._cdc_change_handler.is_alive(),
        ]

        watch_fixture.stop()
        time.sleep(2)
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
        """
        A small test which tests, insert from superduperdp should not execute
        task graph.
        """
        watch_fixture._cdc_change_handler = MagicMock()
        watch_fixture.watch()
        output, _ = with_vector_index.execute(
            Collection(name='documents').insert_many(an_insert, refresh=True)
        )

        _id = output.inserted_ids[0]
        doc = with_vector_index.db['documents'].find_one({'_id': _id})
        watch_fixture.stop()
        assert '_outputs' not in doc.keys()
