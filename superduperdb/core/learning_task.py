from typing import List, Mapping, Optional

from superduperdb.core.base import Component, ComponentList, PlaceholderList, Placeholder
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.datalayer.base.query import Select


class LearningTask(Component):
    """
    Learning-task base object, used to hold important object crucial to all learning-task
    creation and management.

    :param identifier: Unique identifier of learning task
    :param keys: Keys - keys or columns to which to apply the models
    :param select: Select data to apply learning to
    :param validation_sets: List of validation-datasets
    :param training_configuration_id: Identifier of a training configuration
    :param training_configuration: A training configuration instance
    :param metrics: List of Metric components to measure performance on validation-sets
    :param metric_ids: List of identifiers of metrics
    :param models: List of Model components
    :param model_ids: List of Model identifiers
    :param features: Dictionary of feature mappings from keys -> model-identifiers
    """
    variety = 'learning_task'

    def __init__(
        self,
        identifier: str,
        keys: List[str],
        select: Select,
        validation_sets: List[str] = (),
        training_configuration_id: Optional[str] = None,
        training_configuration: Optional[TrainingConfiguration] = None,
        metrics: Optional[List[Metric]] = None,
        metric_ids: Optional[List[str]] = None,
        models: Optional[List[Model]] = None,
        model_ids: Optional[List[str]] = None,
        features: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(identifier)

        assert training_configuration_id or training_configuration
        assert model_ids or models

        self.models = ComponentList('model', models) if models else PlaceholderList('model', model_ids)
        self.keys = keys
        self.training_configuration = (
            training_configuration if training_configuration
            else Placeholder(training_configuration_id, 'training_configuration')
        )
        self.identifier = identifier
        self.validation_sets = validation_sets
        self.metrics = ComponentList('metric', metrics) \
            if metrics else (PlaceholderList('metric', metric_ids) if metric_ids else ())
        self.features = features or {}
        self.select = select

    def asdict(self):
        return {
            'identifier': self.identifier,
            'keys': list(self.keys),
            'validation_sets': list(self.validation_sets),
            'training_configuration': self.training_configuration.identifier,
            'metrics': [m.identifier for m in self.metrics],
            'models': [m.identifier for m in self.models],
            'features': self.features,
        }

    def schedule_jobs(self, database, verbose=True, dependencies=()):
        return [database.fit(self.identifier)]
