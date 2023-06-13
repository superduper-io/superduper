from typing import List, Mapping, Optional, Union

from superduperdb.core.base import (
    Component,
    ComponentList,
    PlaceholderList,
    Placeholder,
    is_placeholders_or_components,
)
from superduperdb.core.metric import Metric
from superduperdb.core.model import Model
from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.datalayer.base.query import Select


class LearningTask(Component):
    """
    Learning-task base object, used to hold important object crucial to all
    learning-task creation and management.

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
        models: Union[List[Model], List[str]],
        keys: List[str],
        select: Select,
        training_configuration: Union[TrainingConfiguration, str],
        validation_sets: List[str] = (),
        metrics: Union[List[Metric], List[str]] = (),
        features: Optional[Mapping[str, str]] = None,
    ):
        super().__init__(identifier)

        models_are_strs, models_are_comps = is_placeholders_or_components(models)
        err_msg = 'Must specify all model IDs or all Model instances directly'
        assert models_are_strs or models_are_comps, err_msg

        if models_are_strs:
            self.models = PlaceholderList('model', models)
        else:
            self.models = ComponentList('model', models)

        err_msg = 'Must specify all metric IDs or all Metric instances directly'
        metrics_are_strs, metrics_are_comps = is_placeholders_or_components(metrics)
        assert metrics_are_strs or metrics_are_comps, err_msg

        if metrics_are_strs:
            self.metrics = PlaceholderList('metric', metrics)
        else:
            self.metrics = ComponentList('metric', metrics)

        self.keys = keys
        self.training_configuration = (
            training_configuration
            if isinstance(training_configuration, TrainingConfiguration)
            else Placeholder(training_configuration, 'training_configuration')
        )
        self.identifier = identifier
        self.validation_sets = validation_sets
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
