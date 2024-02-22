import os
import random
import time

import numpy as np

try:
    import pytest
except ImportError:

    class _pytest:
        def fixture(self, *args, **kwargs):
            return

    pytest = _pytest()

from PIL import Image

from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.components.listener import Listener
from superduperdb.components.model import ObjectModel
from superduperdb.components.schema import Schema
from superduperdb.components.vector_index import VectorIndex
from superduperdb.ext.pillow.encoder import pil_image

# Setup:
# Take empty database
# Add two models
# Add encoder
# create custom serializer
# take output from model 1 as input to model 2
# create listener for model1 and model2
# add new data to collection
# delete data on collection
# start cdc service
# add new data to collection
# collection should have one image, one string, one float, one int data
# Make the entire service as distributed on ray


class Model1:
    def predict(self, x):
        return [
            {
                'image': Image.fromarray(np.random.randn(10, 10, 3).astype(np.uint8)),
                'int': np.asarray([x * 100] * 10),
            }
        ] * random.randint(1, 2)


class Model2:
    def predict(self, x):
        if isinstance(x, str):
            return np.asarray([int(x) * 100] * 10)
        return x


def _wait_for_keys(db, collection='_outputs.int.model1', n=10, key=''):
    retry_left = 5

    def check_outputs():
        docs = list(db.databackend.db[collection].find({}))
        p = 0
        for d in docs:
            try:
                for k in key.split('.'):
                    d = d[k]
            except KeyError:
                pass
            else:
                p += 1
        return len(docs) == p

    while not check_outputs() and retry_left != 0:
        time.sleep(2)
        retry_left -= 1


def _wait_for_outputs(db, collection='_outputs.int.model1', n=10):
    retry_left = 5

    def check_outputs():
        docs = list(db.databackend.db[collection].find({}))
        return docs

    while not len(check_outputs()) >= n and retry_left != 0:
        time.sleep(2)
        retry_left -= 1
    return len(check_outputs())


@pytest.fixture
def distributed_db(monkeypatch, test_db, ray_client):
    from superduperdb import CFG

    cdc = os.environ['SUPERDUPERDB_CLUSTER_CDC_URI']
    vector_search = os.environ['SUPERDUPERDB_CLUSTER_VECTOR_SEARCH']
    monkeypatch.setattr(CFG.cluster.cdc, 'uri', cdc)
    monkeypatch.setattr(CFG.cluster, 'vector_search', vector_search)

    test_db.set_compute(ray_client)

    def update_syspath():
        import sys

        sys.path.append('./')

    test_db.get_compute().submit(update_syspath)

    yield test_db
    from superduperdb.base.config import Cluster

    monkeypatch.setattr(CFG, 'cluster', Cluster())


def test_advance_setup(distributed_db, image_url):
    db = distributed_db
    # Take empty database

    image = [
        {
            '_content': {
                'uri': image_url,
                'datatype': 'pil_image',
                'leaf_type': 'encodable',
            }
        }
    ]

    raw_bytes = 'some raw bytes'.encode('utf-8')
    data = [
        Document(
            {
                'image': image,
                'int': i,
                'text': str(i),
                'float': float(i),
                'raw_bytes': raw_bytes,
            }
        )
        for i in range(5)
    ]

    mixed_input = Collection('mixed_input')
    db.add(pil_image)

    from superduperdb.ext.numpy import array

    model1 = ObjectModel(
        identifier='model1',
        object=Model1().predict,
        flatten=True,
        model_update_kwargs={'document_embedded': False},
        output_schema=Schema(
            identifier='myschema',
            fields={'image': pil_image, 'int': array('int64', (10,))},
        ),
    )
    db.add(model1)

    e = array('int64', (10,))

    model2 = ObjectModel(
        identifier='model2',
        object=Model2().predict,
        datatype=e,
    )

    db.add(model2)

    listener1 = Listener(
        model=db.load('model', 'model1'), key='int', select=mixed_input.find()
    )
    db.add(listener1)

    db.execute(Collection('mixed_input').insert_many(data))

    n = _wait_for_outputs(db=db, n=15)

    db.add(
        VectorIndex(
            identifier='test_search_index',
            measure='l2',
            indexing_listener=Listener(
                model=model2,
                key='_outputs.int.model1.0.int',
                select=Collection('_outputs.int.model1').find(),
            ),
            compatible_listener=Listener(
                model=model2, key='text', select=None, active=False
            ),
        )
    )

    search_phrase = '4'
    _wait_for_keys(
        db=db, n=n, collection='_outputs.int.model1', key='_outputs.int.model2'
    )

    r = next(
        db.execute(
            Collection('_outputs.int.model1')
            .like(
                Document({'text': search_phrase}), vector_index='test_search_index', n=1
            )
            .find()
        )
    )
    r = r.unpack()

    assert '_outputs' in r
    assert np.allclose(
        np.asarray([400] * 10), r['_outputs']['int']['model1']['0']['int']
    )
