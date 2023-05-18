from superduperdb.training.torch.trainer import TorchTrainerConfiguration
from superduperdb.training.validation import validate_semantic_index
from superduperdb.vector_search.vanilla.hashes import VanillaHashSet
from tests.fixtures.collection import (
    n_data_points,
)
import PIL.PngImagePlugin
import os
import pytest
import torch

from tests.material.losses import ranking_loss
from superduperdb.vector_search.vanilla.measures import css

remote = os.environ.get('SUPERDUPERDB_REMOTE_TEST', 'local')
image_url = 'https://www.superduperdb.com/logos/white.png'

remote_values = {
    'local': [False],
    'local+remote': [False, True],
    'remote': [True],
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
    s = with_semantic_index.find_one(
        like={'x': r['x']}, semantic_index='test_learning_task'
    )
    assert s['_id'] == r['_id']


@pytest.mark.parametrize('remote', remote_values)
def test_find_faiss(with_semantic_index, remote):
    with_semantic_index.remote = remote
    print(with_semantic_index.count_documents({}))
    r = with_semantic_index.find_one()
    s = with_semantic_index.find_one(
        like={'x': r['x']},
        semantic_index='test_learning_task',
        hash_set_cls='faiss',
    )
    assert s['_id'] == r['_id']


@pytest.mark.parametrize('remote', remote_values)
def test_insert(random_data, a_watcher, an_update, remote):
    random_data.remote = remote
    _, task_workflow = random_data.insert_many(an_update)
    if remote:
        for n in task_workflow.G.nodes:
            n = task_workflow.G.nodes[n]
            n['future'].result()
    assert random_data.count_documents({}) == n_data_points + 10


@pytest.mark.parametrize('remote', remote_values)
def test_insert_from_uris(empty, image_type, remote):
    empty.remote = remote
    to_insert = [
        {
            'item': {
                '_content': {
                    'uri': image_url,
                    'type': 'image',
                }
            },
            'other': {
                'item': {
                    '_content': {
                        'uri': image_url,
                        'type': 'image',
                    }
                }
            },
        }
        for _ in range(2)
    ]
    output, G = empty.insert_many(to_insert)
    if remote:
        for node in G.G.nodes:
            empty.watch_job(G.G.nodes[node]['job_id'])
    assert isinstance(empty.find_one()['item'], PIL.PngImagePlugin.PngImageFile)
    assert isinstance(
        empty.find_one()['other']['item'], PIL.PngImagePlugin.PngImageFile
    )


@pytest.mark.parametrize('remote', remote_values)
def test_watcher(random_data, a_model, b_model, remote):
    random_data.remote = remote
    if remote:
        job_id = random_data.create_watcher('linear_a', key='x').key
        random_data.watch_job(job_id)
    else:
        random_data.create_watcher('linear_a', key='x')

    assert 'linear_a' in random_data.find_one()['_outputs']['x']

    outputs = random_data.insert_many(
        [{'x': torch.randn(32), 'update': True} for _ in range(5)]
    )
    if remote:
        for node in outputs[1].G.nodes:
            random_data.watch_job(outputs[1].G.nodes[node]['job_id'])

    assert 'linear_a' in random_data.find_one({'update': True})['_outputs']['x']

    if remote:
        job_id = random_data.create_watcher(
            'linear_b', key='x', features={'x': 'linear_a'}
        ).key
        random_data.watch_job(job_id)
    else:
        random_data.create_watcher(
            'linear_b', key='x', features={'x': 'linear_a'}
        )
    assert 'linear_b' in random_data.find_one()['_outputs']['x']


@pytest.mark.parametrize('remote', remote_values)
def test_learning_task(si_validation, a_model, c_model, metric, remote):
    si_validation.remote = remote
    jobs = si_validation.create_learning_task(
        ['linear_a', 'linear_c'],
        ['x', 'z'],
        identifier='my_index',
        metrics=['p_at_1'],
        configuration=TorchTrainerConfiguration(
            objective=ranking_loss,
            n_iterations=4,
            validation_interval=20,
            loader_kwargs={'batch_size': 10, 'num_workers': 0},
            optimizer_classes={
                'linear_a': torch.optim.Adam,
                'linear_c': torch.optim.Adam,
            },
            optimizer_kwargs={
                'linear_a': {'lr': 0.001},
                'linear_c': {'lr': 0.001},
            },
            compute_metrics=validate_semantic_index,
            hash_set_cls=VanillaHashSet,
            measure=css,
        ),
        validation_sets=('my_valid',),
    )

    if remote:
        for job in jobs:
            si_validation.watch_job(job.key)


@pytest.mark.parametrize('remote', remote_values)
def test_predict(a_model, remote):
    a_model.remote = remote
    a_model.predict_one('linear_a', torch.randn(32))
