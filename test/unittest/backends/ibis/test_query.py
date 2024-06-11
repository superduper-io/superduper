from test.db_config import DBConfig

import numpy
import pytest

from superduperdb.backends.ibis.field_types import dtype
from superduperdb.base.document import Document
from superduperdb.components.schema import Schema
from superduperdb.components.table import Table
from superduperdb.ext.pillow.encoder import pil_image

try:
    import torch
except ImportError:
    torch = None


def test_serialize_table():
    schema = Schema(
        identifier='my_schema',
        fields={
            'id': dtype('int64'),
            'health': dtype('int32'),
            'age': dtype('int32'),
            'image': pil_image,
        },
    )

    s = schema.encode()
    ds = Document.decode(s).unpack()
    assert isinstance(ds, Schema)

    t = Table(identifier='my_table', schema=schema)

    s = t.encode()

    ds = Document.decode(s).unpack()
    assert isinstance(ds, Table)


# TODO: Do we need this test?
@pytest.mark.skip
def test_auto_inference_primary_id():
    s = Table('other', primary_id='other_id')
    t = Table('test', primary_id='id')

    q = t.join(s, t.id == s.other_id)

    assert q.primary_id == ['id', 'other_id']

    q = t.join(s, t.id == s.other_id).group_by('id')

    assert q.primary_id == 'other_id'


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (
            DBConfig.sqldb_data,
            {'n_data': 5, 'add_vector_index': True, 'add_models': True},
        ),
    ],
    indirect=True,
)
def test_renamings(db):
    t = db['documents']
    listener_uuid = [k.split('/')[-1] for k in db.show('listener')][0]
    q = t.select('id', 'x', 'y').outputs(listener_uuid)
    data = list(db.execute(q))
    assert torch.is_tensor(data[0].unpack()[f'_outputs.{listener_uuid}'])


@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.sqldb_empty),
    ],
    indirect=True,
)
def test_serialize_query(db):
    from superduperdb.backends.ibis.query import IbisQuery

    t = IbisQuery(db=db, table='documents', parts=[('select', ('id',), {})])

    q = t.filter(t.id == 1).select(t.id, t.x)

    print(Document.decode(q.encode()).unpack())


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.sqldb_data, {'n_data': 10}),
    ],
    indirect=True,
)
def test_add_fold(db):
    table = db['documents']
    select_train = table.select('id', 'x', '_fold').add_fold('train')
    result_train = db.execute(select_train)

    select_valid = table.select('id', 'x', '_fold').add_fold('valid')
    result_valid = db.execute(select_valid)
    result_train = list(result_train)
    result_valid = list(result_valid)
    assert len(result_train) + len(result_valid) == 10


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.sqldb_data, {'n_data': 500}),
    ],
    indirect=True,
)
def test_get_data(db):
    db['documents'].limit(2)
    db.metadata.get_component('table', 'documents')


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.sqldb_data, {'n_data': 5}),
    ],
    indirect=True,
)
def test_insert_select(db):
    q = db['documents'].select('id', 'x', 'y').limit(2)
    r = list(db.execute(q))

    assert len(r) == 2
    assert all(all([k in ['id', 'x', 'y'] for k in x.unpack().keys()]) for x in r)


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (DBConfig.sqldb_data, {'n_data': 5}),
    ],
    indirect=True,
)
def test_filter(db):
    t = db['documents']
    q = t.select('id', 'y')
    r = list(db.execute(q))
    ys = [x['y'] for x in r]
    uq = numpy.unique(ys, return_counts=True)

    q = t.select('id', 'y').filter(t.y == uq[0][0])
    r = list(db.execute(q))
    assert len(r) == uq[1][0]


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (
            DBConfig.sqldb_data,
            {'n_data': 5, 'add_vector_index': True, 'add_models': True},
        ),
    ],
    indirect=True,
)
def test_pre_like(db):
    r = list(db.execute(db['documents'].select('id', 'x')))[0]
    query = (
        db['documents']
        .like(
            r=Document({'x': r['x']}),
            vector_index='test_vector_search',
        )
        .select('id')
    )
    s = list(db.execute(query))[0]
    assert r['id'] == s['id']


@pytest.mark.skipif(not torch, reason='Torch not installed')
@pytest.mark.parametrize(
    "db",
    [
        (
            DBConfig.sqldb_data,
            {'n_data': 5, 'add_vector_index': True, 'add_models': True},
        ),
    ],
    indirect=True,
)
def test_post_like(db):
    r = list(db.execute(db['documents'].select('id', 'x')))[0]
    t = db['documents']
    query = (
        t.select('id', 'x', 'y')
        .filter(t.id.isin(['1', '2', '3']))
        .like(
            r=Document({'x': r['x']}),
            vector_index='test_vector_search',
        )
        .limit(2)
    )
    s = list(db.execute(query))
    assert len(s) == 2
    assert all([d['id'] in ['1', '2', '3'] for d in s])
