from tests.fixtures.collection import empty, float_tensors, random_data

from superduperdb.training.loading import QueryDataset


def test_query_dataset(random_data):

    train_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}),  # can also include projection as third parameter
        fold='train',
    )

    print(len(train_data))
    print(train_data[0])

    valid_data = QueryDataset(
        database_type='mongodb',
        database='test_db',
        query_params=('documents', {}),  # can also include projection as third parameter
        fold='valid',
    )

    print(len(valid_data))
