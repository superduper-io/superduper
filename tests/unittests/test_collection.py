from tests.fixtures.collection import (
    float_tensors, empty, a_model, b_model, c_model, random_data,
    si_validation, measure, metric, my_rank_obj, a_classifier, a_target, accuracy_metric,
    my_class_obj, imputation_validation, with_semantic_index, an_update, a_watcher, image_type,
    n_data_points, with_faiss_semantic_index
)
import PIL.PngImagePlugin
import os
import pytest
import torch


remote = os.environ.get('SUPERDUPERDB_REMOTE_TEST', 'local')
image_url = 'https://www.superduperdb.com/logos/white.png'

remote_values = {
    'local': [False],
    'local+remote': [False, True],
    'remote': [True]
}[remote]


@pytest.mark.parametrize('remote', remote_values)
def test_get_data(random_data, remote):
    r = random_data.find_one()
    print(r)


@pytest.mark.parametrize('remote', remote_values)
def test_find(with_semantic_index, remote):
    with_semantic_index.remote = remote
    print(with_semantic_index.count_documents({}))
    r = with_semantic_index.find_one()
    s = with_semantic_index.find_one(like={'x': r['x']})
    assert s['_id'] == r['_id']


@pytest.mark.parametrize('remote', remote_values)
def test_find_faiss(with_faiss_semantic_index, remote):
    with_faiss_semantic_index.remote = remote
    print(with_faiss_semantic_index.count_documents({}))
    h = with_faiss_semantic_index._all_hash_sets['test_semantic_index']
    r = with_faiss_semantic_index.find_one()
    s = with_faiss_semantic_index.find_one(like={'x': r['x']})
    assert s['_id'] == r['_id']


@pytest.mark.parametrize('remote', remote_values)
def test_insert(random_data, a_watcher, an_update, remote):
    random_data.remote = remote
    _, G = random_data.insert_many(an_update)
    if remote:
        print(G)
        for node in G.nodes:
            node = G.nodes[node]
            random_data.watch_job(node['job_id'])
    assert random_data.count_documents({}) == n_data_points + 10


@pytest.mark.parametrize('remote', remote_values)
def test_insert_from_urls(empty, image_type, remote):
    empty.remote = remote
    to_insert = [
        {
            'item': {
                '_content': {
                    'url': image_url,
                    'type': 'image',
                }
            },
            'other': {
                'item': {
                    '_content': {
                        'url': image_url,
                        'type': 'image',
                    }
                }
            }
        }
        for _ in range(2)
    ]
    output = empty.insert_many(to_insert)
    if remote:
        jobs = output[1]
        for node in jobs:
            for job in jobs[node]:
                empty.watch_job(job)
    assert isinstance(empty.find_one()['item'], PIL.PngImagePlugin.PngImageFile)
    assert isinstance(empty.find_one()['other']['item'], PIL.PngImagePlugin.PngImageFile)


@pytest.mark.parametrize('remote', remote_values)
def test_watcher(random_data, a_model, b_model, remote):

    random_data.remote = remote
    job_id = random_data.create_watcher('linear_a/x', 'linear_a', key='x')
    if remote:
        random_data.watch_job(job_id)

    assert 'linear_a' in random_data.find_one()['_outputs']['x']

    outputs = random_data.insert_many([{'x': torch.randn(32), 'update': True}
                                      for _ in range(5)])
    if remote:
        for node in outputs[1]:
            for job_id in outputs[1][node]:
                random_data.watch_job(job_id)

    assert 'linear_a' in random_data.find_one({'update': True})['_outputs']['x']

    job_id = random_data.create_watcher('linear_b/x', 'linear_b', key='x', features={'x': 'linear_a'})
    if remote:
        random_data.watch_job(job_id)
    assert 'linear_b' in random_data.find_one()['_outputs']['x']


@pytest.mark.parametrize('remote', remote_values)
def test_semantic_index(si_validation, a_model, c_model, measure, metric, my_rank_obj,
                        remote):

    si_validation.remote = remote
    jobs = si_validation.create_semantic_index('my_index',
                                               ['linear_a', 'linear_c'],
                                               ['x', 'z'],
                                               'css',
                                               metrics=['p_at_1'],
                                               objective='rank_obj',
                                               validation_sets=('my_valid',),
                                               trainer_kwargs={'n_iterations': 4, 'validation_interval': 2})
    if remote:
        for job_id in jobs:
            si_validation.watch_job(job_id)


@pytest.mark.parametrize('remote', remote_values)
def test_imputation(imputation_validation, a_classifier, a_target, my_class_obj,
                           accuracy_metric, remote):

    imputation_validation.remote = remote
    jobs = imputation_validation.create_imputation('my_imputation',
                                                   'classifier',
                                                   'x',
                                                   'target',
                                                   'y',
                                                   metrics=['accuracy_metric'],
                                                   objective='class_obj',
                                                   validation_sets=('my_imputation_valid',),
                                                   trainer_kwargs={'n_iterations': 4, 'validation_interval': 2})
    if remote:
        for job_id in jobs:
            imputation_validation.watch_job(job_id)


@pytest.mark.parametrize('remote', remote_values)
def test_apply_model(a_model, remote):
    a_model.remote = remote
    out = a_model.apply_model('linear_a', torch.randn(32))
    print(out)
