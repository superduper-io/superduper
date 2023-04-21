from tests.fixtures.collection import empty
from superduperdb.jobs.graph import TaskWorkflow


def test_graph(empty):

    w = TaskWorkflow(empty.database)

    w.add_node(
        'train(1234)',
        data={
            'task': empty.database.train,
            'args': ['1234', 'imputation'],
            'kwargs': {},
        })

    w.add_node(
        'train(5678)',
        data={
            'task': empty.database.train,
            'args': ['5678', 'imputation'],
            'kwargs': {},
        })

    w.add_edge('train(1234)', 'train(5678)')

    print(w.compile(dry_run=True))
