import os

import PIL.Image
import pytest
import torch.nn
import torchvision

from superduperdb import CFG
from superduperdb.backends.ibis.field_types import dtype
from superduperdb.backends.ibis.query import IbisQuery, RawSQL
from superduperdb.base.document import Document as D
from superduperdb.components.listener import Listener
from superduperdb.components.schema import Schema
from superduperdb.ext.pillow.encoder import pil_image
from superduperdb.ext.torch.encoder import tensor
from superduperdb.ext.torch.model import TorchModel

DO_SKIP = CFG.data_backend.startswith('mongo')


from superduperdb import superduper


@pytest.fixture
def clean_cache():
    directory = './deploy/testenv/cache'

    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


@pytest.mark.skipif(DO_SKIP, reason="skipping ibis tests if mongodb")
def test_end_2_end(clean_cache):
    memory_table = False
    if CFG.data_backend.startswith('duckdb') or CFG.data_backend.endswith('csv'):
        memory_table = True
    _end_2_end(superduper(), memory_table=memory_table)


def _end_2_end(db, memory_table=False):
    schema = Schema(
        identifier='my_table',
        fields={
            'id': dtype('str'),
            'health': dtype('int32'),
            'age': dtype('int32'),
            'image': pil_image,
        },
    )
    im = PIL.Image.open('test/material/data/test-image.jpeg')

    data_to_insert = [
        {'id': '1', 'health': 0, 'age': 25, 'image': im},
        {'id': '2', 'health': 1, 'age': 26, 'image': im},
        {'id': '3', 'health': 0, 'age': 27, 'image': im},
        {'id': '4', 'health': 1, 'age': 28, 'image': im},
    ]

    from superduperdb.components.table import Table
    t = Table(identifier='my_table', schema=schema, db=db)

    db.add(t)
    t = db['my_table']

    insert = t.insert(
        [
            D(
                {
                    'id': d['id'],
                    'health': d['health'],
                    'age': d['age'],
                    'image': d['image'],
                }
            )
            for d in data_to_insert
        ]
    )
    db.execute(insert)

    q = t.select('image', 'age', 'health')

    result = db.execute(q)
    for img in result:
        img = img.unpack()
        assert isinstance(img['image'], PIL.Image.Image)
        assert isinstance(img['age'], int)
        assert isinstance(img['health'], int)

    # preprocessing function
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    def postprocess(x):
        return int(x.topk(1)[1].item())

    # create a torchvision model
    resnet = TorchModel(
        identifier='resnet18',
        preprocess=preprocess,
        postprocess=postprocess,
        object=torchvision.models.resnet18(pretrained=False),
        datatype=dtype('int32'),
    )

    # Apply the torchvision model
    listener1 = Listener(
        model=resnet,
        key='image',
        select=t.select('id', 'image'),
        predict_kwargs={'max_chunk_size': 3000},
        identifier='listener1',
    )
    db.add(listener1)

    # also add a vectorizing model
    vectorize = TorchModel(
        preprocess=lambda x: torch.randn(32),
        object=torch.nn.Linear(32, 16),
        identifier='model_linear_a',
        datatype=tensor(dtype='float', shape=(16,)),
    )

    # create outputs query
    q = t.outputs(listener1.predict_id)

    # apply to the table
    listener2 = Listener(
        model=vectorize,
        key=listener1.outputs,
        select=q,
        predict_kwargs={'max_chunk_size': 3000},
        identifier='listener2',
    )
    db.add(listener2)

    # Build query to get the results back
    q = t.outputs(listener2.outputs).select('id', 'image', 'age').filter(t.age > 25)

    # Get the results
    result = list(db.execute(q))
    assert result
    assert 'image' in result[0].unpack()

    # TODO: Make this work
    '''

    q = t.select('id', 'image', 'age').filter(t.age > 25).outputs(listener2.outputs)

    # Get the results
    result = list(db.execute(q))
    assert listener2.outputs in result[0].unpack()
    '''

    # Raw query
    # TODO: Support RawSQL
    '''

    if not memory_table:
        query = RawSQL(query='SELECT id from my_table')
        rows = list(db.execute(query))
        assert 'id' in list(rows[0].unpack().keys())
        assert [r.unpack() for r in rows] == [
            {'id': '1'},
            {'id': '2'},
            {'id': '3'},
            {'id': '4'},
        ]
    '''


@pytest.mark.skipif(DO_SKIP, reason="skipping ibis tests if mongodb")
def test_nested_query(clean_cache):
    db = superduper()

    memory_table = False
    if CFG.data_backend.endswith('csv'):
        memory_table = True
    schema = Schema(
        identifier='my_table',
        fields={
            'id': dtype('int64'),
            'health': dtype('int32'),
            'age': dtype('int32'),
            'image': pil_image,
        },
    )

    from superduperdb.components.table import Table
    t = Table(identifier='my_table', schema=schema)

    db.add(t)

    t = db['my_table']
    q = t.filter(t.age >= 10)

    expr_ = q.compile(db)

    if not memory_table:
        assert 'SELECT t0._fold, t0.id, t0.health, t0.age, t0.image' in str(
            expr_
        )
    else:
        assert 'Selection[r0]\n  predicates:\n    r0.age >= 10' in str(expr_)
        assert (
            'my_table\n  id     int64\n  _fold  string\n  health int32\n  age    '
            'int32\n  image  binary' in str(expr_)
        )
