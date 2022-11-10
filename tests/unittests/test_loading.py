import torch
import torch.utils.data

from sddb import cf
from sddb.training.loading import QueryDataset

from tests.fixtures.collection import collection_hashes


def test_query_dataset(collection_hashes):
    ds = QueryDataset(
        client=cf['mongodb'],
        database='test_db',
        collection='test_collection',
    )
    assert isinstance(ds[0]['_outputs']['_base']['dummy'], torch.Tensor)
    assert len(ds) == 10

    loader = torch.utils.data.DataLoader(ds, batch_size=10)
    for b in loader:
        break

    assert len(b['_outputs']['_base']['dummy'].shape) == 2
    assert b['_outputs']['_base']['dummy'].shape[0] == 10
