from tests.fixtures.collection import random_data, a_model, float_tensors, empty


def test_submit(random_data, a_model):
    random_data.remote = True
    result = random_data.database.apply_watcher(
        'none',
        watcher_info={'model': 'linear_a', 'query_params': ['documents', {}], 'key': 'x'},
    )
    print(result.result())
    r = random_data.find_one()
    print(r)