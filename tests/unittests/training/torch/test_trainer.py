# ruff: noqa: F401, F811

from superduperdb.training.torch.trainer import TorchTrainerConfiguration
from superduperdb.training.validation import validate_semantic_index
from tests.material.losses import ranking_loss
from tests.material.metrics import PatK

from superduperdb.vector_search.vanilla.measures import dot
from superduperdb.vector_search.vanilla.hashes import VanillaHashSet

from tests.fixtures.collection import (
    random_data,
    float_tensors,
    empty,
    a_model,
    c_model,
)


def test_semantic_index(random_data, a_model, c_model):
    cf = TorchTrainerConfiguration(
        objective=ranking_loss,
        optimizer_kwargs={'lr': 0.0001},
        loader_kwargs={'batch_size': 100, 'num_workers': 0},
        no_improve_then_stop=3,
        max_iterations=20,
        compute_metrics=validate_semantic_index,
        hash_set_cls=VanillaHashSet,
        measure=dot,
    )

    random_data.create_validation_set('test_validation')
    random_data.create_metric('p_at_1', PatK(1))
    random_data.create_learning_task(
        ['linear_a', 'linear_c'],
        ['x', 'z'],
        configuration=cf,
        metrics=['p_at_1'],
        validation_sets=('test_validation',),
    )
