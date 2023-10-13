import random
import time

import pytest
from superduperdb.misc.runnable.collection import ThreadQueue


@pytest.mark.parametrize('thread_count', [1, 3])
def test_thread_queue(thread_count):
    result = []

    def callback(item):
        result.append(item)
        time.sleep(random.uniform(0.001, 0.010))

    with ThreadQueue(callback, thread_count=thread_count) as tq:
        for i in range(8):
            tq.queue.put(i)

    assert not tq.running
    assert sorted(result) == list(range(8))


@pytest.mark.parametrize('thread_count', [1, 3])
def test_thread_queue_error(thread_count):
    result = []
    errors = []

    def cb(item):
        result.append(item)
        if len(result) == 5:
            result.append('ERROR')
            raise ValueError('An', 'error')

        time.sleep(random.uniform(0.001, 0.010))

    with ThreadQueue(callback=cb, error=errors.append, thread_count=thread_count) as tq:
        for i in range(8):
            tq.queue.put(i)

    assert not tq.running
    assert result == (list(range(5)) + ['ERROR'])
    assert [e.args for e in errors] == [('An', 'error')]
