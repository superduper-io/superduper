from test.torch import skip_torch, torch
from unittest import mock

import numpy as np
import pytest

from superduperdb.vector_search.base import BaseVectorIndex, to_numpy


class TestBaseVectorIndex:
    @pytest.fixture
    def base_vector_index(self):
        # Create a sample BaseVectorIndex instance for testing
        h = np.random.rand(10, 3)  # Sample data
        index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']  # Sample index
        measure = 'euclidean'  # Sample measure
        return BaseVectorIndex(h, index, measure)

    def test_init(self, base_vector_index):
        assert isinstance(base_vector_index.h, np.ndarray)
        assert isinstance(base_vector_index.index, list)
        assert isinstance(base_vector_index.lookup, dict)
        assert isinstance(base_vector_index.measure, str)

    def test_shape(self, base_vector_index):
        assert base_vector_index.shape == base_vector_index.h.shape

    @mock.patch('superduperdb.vector_search.base.to_numpy')
    def test_find_nearest_from_id(self, mock_to_numpy, base_vector_index):
        # Mock to_numpy function to avoid its actual execution
        mock_to_numpy.return_value = np.array([[0.1, 0.2, 0.3]])

        _id = 'a'
        n = 100
        with pytest.raises(NotImplementedError):
            base_vector_index.find_nearest_from_id(_id, n)

    @mock.patch('superduperdb.vector_search.base.to_numpy')
    def test_find_nearest_from_ids(self, mock_to_numpy, base_vector_index):
        # Mock to_numpy function to avoid its actual execution
        mock_to_numpy.return_value = np.array([[0.1, 0.2, 0.3]])

        _ids = ['a', 'b']
        n = 100
        with pytest.raises(NotImplementedError):
            base_vector_index.find_nearest_from_ids(_ids, n)

    @mock.patch('superduperdb.vector_search.base.to_numpy')
    def test_find_nearest_from_array(self, mock_to_numpy, base_vector_index):
        # Mock to_numpy function to avoid its actual execution
        mock_to_numpy.return_value = np.array([[0.1, 0.2, 0.3]])

        h = np.random.rand(1, 3)  # Sample array
        n = 100
        with pytest.raises(NotImplementedError):
            base_vector_index.find_nearest_from_array(h, n)

    @mock.patch('superduperdb.vector_search.base.to_numpy')
    def test_find_nearest_from_arrays(self, mock_to_numpy, base_vector_index):
        # Mock to_numpy function to avoid its actual execution
        mock_to_numpy.return_value = np.array([[0.1, 0.2, 0.3]])

        h = np.random.rand(2, 3)  # Sample array
        n = 100
        with pytest.raises(NotImplementedError):
            base_vector_index.find_nearest_from_arrays(h, n)

    def test_getitem(self, base_vector_index):
        with pytest.raises(NotImplementedError):
            base_vector_index['item']


@skip_torch
def test_to_numpy():
    # Test to_numpy function with different input types
    x = np.array([[1, 2, 3], [4, 5, 6]])
    result = to_numpy(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, x)

    x = torch.tensor([[1, 2, 3], [4, 5, 6]])
    result = to_numpy(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, x.numpy())

    x = [[1, 2, 3], [4, 5, 6]]
    result = to_numpy(x)
    assert isinstance(result, np.ndarray)
    assert np.array_equal(result, np.array(x))
