from test.db_config import DBConfig

import pytest

from superduperdb.components.graph import Graph
from superduperdb.components.model import Model


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
        return x + 2

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


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_simple_graph(db, model1, model2):
    g = Graph(identifier='simple-graph', db=db)
    intermediate_node = g.connect(g, model1)
    g.connect(model1, model2)
    assert g.predict(1, one=True) == 4
    assert intermediate_node.output == 2

    # with names
    g = Graph(identifier='simple-graph-name', db=db)
    g.connect(g, 'm1')
    g.connect('m1', 'm2')
    assert g.predict(1, one=True) == 4

    g = Graph(identifier='simple-graph', db=db)
    intermediate_node = g.connect(g, model1)
    g.connect(model1, model2)
    assert g.predict([1, 2, 3], one=False) == [4, 5, 6]


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_complex_graph(db, model1, model2_multi, model3):
    db = db
    g = Graph(identifier='complex-graph', db=db)
    g.connect(g, model1)
    g.connect(g, model2_multi, on='x')
    g.connect(model1, model2_multi, on='y')
    g.connect(model1, model3, on='x')
    g.connect(model2_multi, model3, on='y')
    assert g.predict(1, one=True) == 10

    assert g.predict([1, 2, 3], one=False) == [10, 13, 16]


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_non_dag(db, model1, model2):
    db = db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(g, model1)
        g.connect(model1, model2)
        g.connect(model2, model1)
        assert 'The graph is not DAG' in str(excinfo.value)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_disconnected_edge(db, model1, model2_multi):
    db = db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(g, model1)
        g.connect(model1, model2_multi, on='x')
        g.predict(1)
        assert 'Graph disconnected at Node: m2_multi' in str(excinfo.value)


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_no_root(db, model1, model2_multi):
    db = db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(model1, model2_multi, on='x')
        g.predict(1)
        assert 'Root graph node is not present' in str(excinfo.value)


def test_complex_graph_with_select(db):
    g = Graph(identifier='complex-graph', db=db)
    g.connect(g, 'linear_a')
    g.connect('linear_a', 'linear_b')

    from superduperdb.backends.mongodb import Collection

    select = Collection('documents').find({})
    g.predict(X='x', db=db, select=select)
    assert all(
        ['complex-graph' in x['_outputs']['x'] for x in list(db.execute(select))]
    )
