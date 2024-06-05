from test.db_config import DBConfig

import networkx as nx
import pytest

from superduperdb import ObjectModel
from superduperdb.components.graph import Graph, document_node, input_node


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model1(db):
    def model_object(x):
        return x + 1

    model = ObjectModel(identifier='m1', object=model_object, signature='singleton')
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model2(db):
    def model_object(x):
        return x + 2, x

    model = ObjectModel(identifier='m2', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model2_multi_dict(db):
    def model_object(x):
        return {'x': x + 2}

    model = ObjectModel(identifier='m2_multi_dict', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model2_multi(db):
    def model_object(x, y=1):
        return x + y + 2

    model = ObjectModel(identifier='m2_multi', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model3(db):
    def model_object(x, y):
        return x + y + 3

    model = ObjectModel(identifier='m3', object=model_object)
    db.add(model)
    yield model


def test_simple_graph(model1, model2):
    g = Graph(
        identifier='simple-graph', input=model1, outputs=model2, signature='*args'
    )
    g.connect(model1, model2)
    assert g.predict(1) == (4, 2)

    g = Graph(
        identifier='simple-graph', input=model1, outputs=model2, signature='*args'
    )
    g.connect(model1, model2)
    assert g.predict_batches([1, 2, 3]) == [(4, 2), (5, 3), (6, 4)]


def test_graph_output_indexing(model2_multi_dict, model2, model1):
    g = Graph(
        identifier='simple-graph',
        input=model1,
        outputs=[model2],
        signature='**kwargs',
    )
    g.connect(model1, model2_multi_dict, on=(None, 'x'))
    g.connect(model2_multi_dict, model2, on=('x', 'x'))
    assert g.predict(1) == [(6, 4)]


def test_complex_graph(model1, model2_multi, model3, model2):
    g = Graph(
        identifier='complex-graph',
        input=model1,
        outputs=[model2, model2_multi],
    )
    g.connect(model1, model2_multi, on=(None, 'x'))
    g.connect(model1, model2)
    g.connect(model2, model2_multi, on=(0, 'y'))
    g.connect(model2, model3, on=(1, 'x'))
    g.connect(model2_multi, model3, on=(None, 'y'))
    assert g.predict(1) == [(4, 2), 8]
    assert g.predict_batches([1, 2, 3]) == [
        [(4, 2), (5, 3), (6, 4)],
        [8, 10, 12],
    ]


def test_non_dag(model1, model2):
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', input=model1)
        g.connect(model1, model2)
        g.connect(model2, model1)
        assert 'The graph is not DAG' in str(excinfo.value)


def test_disconnected_edge(model1, model2_multi):
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', input=model1, outputs=[model2_multi])
        g.connect(model1, model2_multi, on=(-1, 'x'))
        g.predict(1)
        assert 'Graph disconnected at Node: m2_multi' in str(excinfo.value)


def test_complex_graph_with_select(db):
    linear_a = db.load('model', 'linear_a')
    linear_b = db.load('model', 'linear_b')
    g = Graph(identifier='complex-graph', input=linear_a, outputs=[linear_b])
    g.connect(linear_a, linear_b)

    select = db["documents"].find({})
    g.predict_in_db(X='x', select=select, db=db, predict_id='test')
    assert all(['test' in x['_outputs'] for x in list(db.execute(select))])


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_serialization(db, model1):
    g = Graph(identifier='complex-graph', input=model1)
    original_g = g.G
    db.add(g)
    g = db.load('model', 'complex-graph')
    assert nx.utils.graphs_equal(original_g, g.G)


def test_to_graph():
    model1 = ObjectModel('model_1', object=lambda x: (x + 1, x + 4))
    model2 = ObjectModel('model_2', object=lambda x, y: (x + 2) * y)
    in_ = input_node('number')
    out1 = model1(x=in_)
    out2 = model2(x=out1[1], y=in_)
    graph = out2.to_graph('my_graph')
    print(graph.predict(5))


def test_to_listeners():
    model1 = ObjectModel('model_1', object=lambda x: x + 1)
    model2 = ObjectModel('model_2', object=lambda x, y: (x + 2) * y)
    in_ = document_node('number')
    output1 = model1(x=in_['number'], outputs='l1')
    output2 = model2(x=output1, y=in_['number'], outputs='l2')
    listener_stack = output2.to_listeners(select=None, identifier='test_to_listeners')
    import pprint

    print('\n')
    pprint.pprint(listener_stack)
