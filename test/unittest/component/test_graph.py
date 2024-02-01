import pytest

from superduperdb.components.graph import Graph
from superduperdb.components.model import Model


@pytest.fixture
def model1(test_db):
    def model_object(x):
        return x + 1

    model = Model(identifier='m1', object=model_object)
    test_db.add(model)
    yield model


@pytest.fixture
def model2(test_db):
    def model_object(x):
        return x + 2

    model = Model(identifier='m2', object=model_object)
    test_db.add(model)
    yield model


@pytest.fixture
def model2_multi(test_db):
    def model_object(x, y):
        return x + y + 2

    model = Model(identifier='m2_multi', object=model_object)
    test_db.add(model)
    yield model


@pytest.fixture
def model3(test_db):
    def model_object(x, y):
        return x + y + 3

    model = Model(identifier='m3', object=model_object)
    test_db.add(model)
    yield model


def test_simple_graph(test_db, model1, model2):
    db = test_db
    g = Graph(identifier='simple-graph', db=db)
    intermediate_node = g.connect(g, model1)
    g.connect(model1, model2)
    assert g.predict(1) == 4
    assert intermediate_node.output == 2

    # with names
    g = Graph(identifier='simple-graph-name', db=db)
    g.connect(g, 'm1')
    g.connect('m1', 'm2')
    assert g.predict(1) == 4


def test_complex_graph(test_db, model1, model2_multi, model3):
    db = test_db
    g = Graph(identifier='complex-graph', db=db)
    g.connect(g, model1)
    g.connect(g, model2_multi, on='x')
    g.connect(model1, model2_multi, on='y')
    g.connect(model1, model3, on='x')
    g.connect(model2_multi, model3, on='y')
    assert g.predict(1) == 10


def test_non_dag(test_db, model1, model2):
    db = test_db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(g, model1)
        g.connect(model1, model2)
        g.connect(model2, model1)
        assert 'The graph is not DAG' in str(excinfo.value)


def test_disconnected_edge(test_db, model1, model2_multi):
    db = test_db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(g, model1)
        g.connect(model1, model2_multi, on='x')
        g.predict(1)
        assert 'Graph disconnected at Node: m2_multi' in str(excinfo.value)


def test_no_root(test_db, model1, model2_multi):
    db = test_db
    with pytest.raises(TypeError) as excinfo:
        g = Graph(identifier='complex-graph', db=db)
        g.connect(model1, model2_multi, on='x')
        g.predict(1)
        assert 'Root graph node is not present' in str(excinfo.value)
