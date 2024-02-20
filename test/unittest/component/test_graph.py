from test.db_config import DBConfig

import networkx as nx
import pytest

from superduperdb.components.graph import Graph
from superduperdb.components.model import Model, Signature


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model1(db):
    def model_object(x):
        return x + 1

    model = Model(identifier='m1', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model2(db):
    def model_object(x):
        return x + 2, x

    model = Model(identifier='m2', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model2_multi(db):
    def model_object(x, y):
        return x + y + 2

    model = Model(identifier='m2_multi', object=model_object)
    db.add(model)
    yield model


@pytest.fixture
@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def model3(db):
    def model_object(x, y):
        return x + y + 3

    model = Model(identifier='m3', object=model_object)
    db.add(model)
    yield model


def test_simple_graph(model1, model2):
    g = Graph(
        identifier='simple-graph', input=model1, outputs=[model2], signature='*args'
    )
    g.connect(model1, model2)
    assert g.predict_one(1) == [(4, 2)]

    g = Graph(
        identifier='simple-graph', input=model1, outputs=[model2], signature='*args'
    )

    g.connect(model1, model2)
    assert g.predict([[1], [2], [3]]) == [[(4, 2), (5, 3), (6, 4)]]


def test_complex_graph(model1, model2_multi, model3, model2):
    g = Graph(
        identifier='complex-graph',
        input=model1,
        outputs=[model2, model2_multi],
        signature=Signature.kwargs,
    )
    g.connect(model1, model2_multi, on=(None, 'x'))
    g.connect(model1, model2)
    g.connect(model2, model2_multi, on=(0, 'y'))
    g.connect(model2, model3, on=(1, 'x'))
    g.connect(model2_multi, model3, on=(None, 'y'))
    assert g.predict_one(1) == [(4, 2), 8]
    assert g.predict([{'x': 1}, {'x': 2}, {'x': 3}]) == [
        [(4, 2), (5, 3), (6, 4)],
        [8, 10, 12],
    ]
    g.signature = Signature.args
    assert g.predict([[1], [2], [3]]) == [[(4, 2), (5, 3), (6, 4)], [8, 10, 12]]


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
        g.predict_one(1)
        assert 'Graph disconnected at Node: m2_multi' in str(excinfo.value)


def test_complex_graph_with_select(db):
    linear_a = db.load('model', 'linear_a')
    linear_b = db.load('model', 'linear_b')
    g = Graph(identifier='complex-graph', input=linear_a, outputs=[linear_b])
    g.connect(linear_a, linear_b)

    from superduperdb.backends.mongodb import Collection

    select = Collection('documents').find({})
    g.predict_in_db(X='x', select=select, db=db)
    assert all(
        ['complex-graph' in x['_outputs']['x'] for x in list(db.execute(select))]
    )


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_serialization(db, model1):
    g = Graph(identifier='complex-graph', input=model1)
    original_g = g.G
    db.add(g)
    g = db.load('graph', 'complex-graph')
    assert nx.utils.graphs_equal(original_g, g.G)
