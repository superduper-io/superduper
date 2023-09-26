import os

import lance
import numpy as np
import pyarrow as pa
import pytest

from superduperdb.vector_search.base import VectorCollectionConfig, VectorCollectionItem
from superduperdb.vector_search.lance import LanceVectorCollection, LanceVectorDatabase

DIMENSIONS = 3


@pytest.fixture
def lance_dataset_write_path(tmp_path):
    'Path to be used for *write* operations'
    return tmp_path / 'test.lance'


@pytest.fixture(scope='session')
def lance_dataset_read_path(tmpdir_factory):
    'Path to be used for *read-only* operations'
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15]]
    typ = pa.list_(pa.field('values', pa.float32(), nullable=False), DIMENSIONS)
    vecs = pa.array([vec for vec in data], type=typ)
    ids = pa.array([str(id_) for id_ in range(len(data))], type=pa.string())
    table = pa.Table.from_arrays([ids, vecs], names=['id', 'vector'])

    path = str(tmpdir_factory.mktemp('lance').join('fake.lance'))
    lance.write_dataset(table, path, mode='create')
    return path


@pytest.fixture
def vector_collection_pair():
    sample = [[1, 2, 3], [4, 5, 6]]
    return [
        VectorCollectionItem(id=str(i), vector=np.array(v))
        for i, v in enumerate(sample)
    ]


@pytest.fixture
def vector_collection_single():
    sample = [[7, 8, 9]]
    return [
        VectorCollectionItem(id=str(2), vector=np.array(v))
        for _, v in enumerate(sample)
    ]


@pytest.fixture
def vector_collection_config(lance_dataset_read_path):
    return VectorCollectionConfig(
        id=f'lance://{lance_dataset_read_path}',
        dimensions=DIMENSIONS,
    )


def test_nonexistent_dataset():
    with pytest.raises(Exception):
        LanceVectorCollection(uri='nonexistent.lance', dimensions=DIMENSIONS)._dataset()


def test_add_single_op(lance_dataset_write_path, vector_collection_pair):
    vc = LanceVectorCollection(uri=lance_dataset_write_path, dimensions=DIMENSIONS)
    vc.add(vector_collection_pair)

    tbl = lance.dataset(lance_dataset_write_path).to_table()
    assert tbl.column('id').to_pylist() == ['0', '1']
    assert tbl.column('vector').to_pylist() == [[1, 2, 3], [4, 5, 6]]


def test_add_multiple_op(
    lance_dataset_write_path, vector_collection_pair, vector_collection_single
):
    vc = LanceVectorCollection(uri=lance_dataset_write_path, dimensions=DIMENSIONS)
    vc.add(vector_collection_pair)
    vc.add(vector_collection_single)

    tbl = lance.dataset(lance_dataset_write_path).to_table()
    assert tbl.column('id').to_pylist() == ['0', '1', '2']
    assert tbl.column('vector').to_pylist() == [[1, 2, 3], [4, 5, 6], [7, 8, 9]]


def test_find_nearest_from_id(lance_dataset_read_path):
    vc = LanceVectorCollection(uri=lance_dataset_read_path, dimensions=DIMENSIONS)
    results = vc.find_nearest_from_id(identifier='0', limit=1)

    assert results[0].id == '0'


def test_find_nearest_from_id_filter_empty(lance_dataset_read_path):
    vc = LanceVectorCollection(uri=lance_dataset_read_path, dimensions=DIMENSIONS)
    results = vc.find_nearest_from_id(identifier='0', limit=1, within_ids=('1', '2'))

    # ``within_ids`` filter applied AFTER nearest neighbors found
    assert results == []


def test_find_nearest_from_id_filter_nonempty(lance_dataset_read_path):
    vc = LanceVectorCollection(uri=lance_dataset_read_path, dimensions=DIMENSIONS)
    results = vc.find_nearest_from_id(identifier='0', limit=2, within_ids=('1', '2'))

    # ``within_ids`` filter applied AFTER nearest neighbors found
    assert len(results) == 1
    assert results[0].id == '1'


def test_find_nearest_from_id_offset(lance_dataset_read_path):
    vc = LanceVectorCollection(uri=lance_dataset_read_path, dimensions=DIMENSIONS)
    results = vc.find_nearest_from_id(identifier='0', limit=2, offset=1)

    # ``offset`` applied AFTER nearest neighbors found
    assert len(results) == 1
    assert results[0].id == '1'


def test_vector_database(vector_collection_config, lance_dataset_read_path):
    root_dir = os.path.dirname(lance_dataset_read_path)
    vd = LanceVectorDatabase(root_dir)
    print(root_dir)

    coll = vd.get_table(vector_collection_config)

    assert coll == vd._collections[vector_collection_config.id]

    assert isinstance(coll, LanceVectorCollection)


def test_vector_database_bad_path(vector_collection_config):
    with pytest.raises(Exception):
        LanceVectorDatabase('nonexistent.lance').get_table(vector_collection_config)


def test_vector_database_no_identifier(lance_dataset_read_path):
    root_dir = os.path.dirname(lance_dataset_read_path)
    conf = VectorCollectionConfig(
        id=lance_dataset_read_path,
        dimensions=DIMENSIONS,
    )
    with pytest.raises(Exception):
        LanceVectorDatabase(root_dir).get_table(conf)
