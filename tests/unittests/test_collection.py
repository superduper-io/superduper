from tests.fixtures.collection import (
    float_tensors, empty, a_model, b_model, c_model, random_data,
    si_validation, measure, metric, my_rank_obj, a_classifier, a_target, accuracy_metric,
    my_class_obj, imputation_validation, with_semantic_index, an_update, a_watcher
)
import pytest
import torch


@pytest.mark.parametrize('remote', [False, True])
def test_find(with_semantic_index, remote):
    with_semantic_index.remote = remote
    r = with_semantic_index.find_one()

    s = with_semantic_index.find_one(like={'x': r['x']})
    assert s['_id'] == r['_id']


@pytest.mark.parametrize('remote', [False, True])
def test_insert(random_data, a_model, an_update, remote):
    random_data.remote = remote
    output = random_data.insert_many(an_update)
    if remote:
        jobs = output[1]
        for node in jobs:
            for job_id in jobs[node]:
                random_data.watch_job(job_id)
    assert random_data.count_documents({}) == 110


def test_watcher(random_data, a_model, b_model):

    random_data.create_watcher('linear_a', key='x')
    assert 'linear_a' in random_data.find_one()['_outputs']['x']

    random_data.insert_many([{'x': torch.randn(32), 'update': True}
                            for _ in range(5)])
    assert 'linear_a' in random_data.find_one({'update': True})['_outputs']['x']

    random_data.create_watcher('linear_b', key='x', features={'x': 'linear_a'})
    assert 'linear_b' in random_data.find_one()['_outputs']['x']


def test_create_semantic_index(si_validation, a_model, c_model, measure, metric, my_rank_obj):

    si_validation.create_semantic_index('my_index',
                                        models=['linear_a', 'linear_c'],
                                        measure='css',
                                        keys=['x', 'z'],
                                        metrics=['p_at_1'],
                                        objective='rank_obj',
                                        validation_sets=('my_valid',),
                                        n_iterations=4,
                                        validation_interval=2)


def test_create_imputation(imputation_validation, a_classifier, a_target, my_class_obj,
                           accuracy_metric):

    imputation_validation.create_imputation('my_imputation',
                                            model='classifier',
                                            model_key='x',
                                            target='target',
                                            target_key='y',
                                            metrics=['accuracy_metric'],
                                            objective='class_obj',
                                            validation_sets=('my_imputation_valid',),
                                            n_iterations=4,
                                            validation_interval=2)

