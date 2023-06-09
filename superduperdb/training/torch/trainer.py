import inspect
from collections import defaultdict

import torch.cuda
import torch.nn
import torch.optim
import torch.utils
from torch.utils.data import DataLoader

from superduperdb.datalayer.base.imports import get_database_from_database_type
from superduperdb.datalayer.base.query import Select
from superduperdb.core.training_configuration import TrainingConfiguration
from superduperdb.misc.special_dicts import ExtensibleDict
from superduperdb.models.torch.utils import to_device, device_of
from superduperdb.training.query_dataset import QueryDataset
from superduperdb.misc.logger import logging


def _default_optimizer():
    return torch.optim.Adam


def _default_kwargs():
    return {'lr': 0.0001}


class TorchTrainerConfiguration(TrainingConfiguration):
    def __init__(
        self,
        identifier,
        objective,
        loader_kwargs,
        optimizer_classes=None,
        optimizer_kwargs=None,
        max_iterations=float('inf'),
        no_improve_then_stop=5,
        splitter=None,
        download=False,
        validation_interval=100,
        watch='objective',
        **kwargs,
    ):
        _optimizer_classes = optimizer_classes or {}
        optimizer_classes = defaultdict(_default_optimizer)
        optimizer_classes.update(_optimizer_classes)
        _optimizer_kwargs = optimizer_kwargs or {}
        optimizer_kwargs = defaultdict(_default_kwargs)
        optimizer_kwargs.update(_optimizer_kwargs)
        super().__init__(
            identifier,
            loader_kwargs=loader_kwargs,
            objective=objective,
            optimizer_classes=optimizer_classes or {},
            optimizer_kwargs=optimizer_kwargs or {},
            no_improve_then_stop=no_improve_then_stop,
            max_iterations=max_iterations,
            splitter=splitter,
            download=download,
            validation_interval=validation_interval,
            watch=watch,
            **kwargs,
        )

    @classmethod
    def split_and_preprocess(cls, sample, models, keys, splitter=None):
        if splitter is not None:
            sample = [r[k] for r, k in zip(splitter(sample), keys)]
        else:
            sample = [sample[k] for k in keys]

        out = []
        for s, m in zip(sample, models):
            try:
                preprocess = m.preprocess
            except AttributeError:

                def preprocess(x):
                    return x

            out.append(preprocess(s))
        return out

    @classmethod
    def saving_criterion(cls, metrics, watch='objective'):
        if watch == 'objective':
            to_watch = [-x for x in metrics['objective']]
        else:
            to_watch = metrics[watch]

        if all([to_watch[-1] >= x for x in to_watch[:-1]]):
            return True

    @classmethod
    def stopping_criterion(
        cls,
        metrics,
        iterations,
        no_improve_then_stop=None,
        watch='objective',
        max_iterations=None,
    ):
        if isinstance(max_iterations, int) and iterations >= max_iterations:
            return True

        if isinstance(no_improve_then_stop, int):
            if watch == 'objective':
                to_watch = [-x for x in metrics['objective']]
            else:
                to_watch = metrics[watch]

            if max(to_watch[-no_improve_then_stop:]) < max(to_watch):
                logging.info('early stopping triggered!')
                return True

        return False

    @classmethod
    def get_validation_dataset(cls, database_type, database_name, validation_set):
        database = get_database_from_database_type(database_type, database_name)
        select: Select = database.get_query_for_validation_set(validation_set)
        return QueryDataset(select, database_name, database_type, fold='valid')

    def __call__(
        self,
        identifier,
        models,
        keys,
        model_names,
        database_type,
        database_name,
        select: Select,
        validation_sets=(),
        metrics=None,
        features=None,
        download=False,
    ):
        database = get_database_from_database_type(database_type, database_name)

        lookup = dict(zip(model_names, models))
        optimizer_classes = defaultdict(lambda: torch.optim.Adam)
        optimizer_classes.update(self.optimizer_classes)
        optimizers = []
        for k in optimizer_classes:
            optimizers.append(
                optimizer_classes[k](lookup[k].parameters(), **self.optimizer_kwargs[k])
            )

        def transform(x):
            return self.split_and_preprocess(x, models, keys, self.splitter)

        train_data, valid_data = self._get_data(
            database_type=database_type,
            database_name=database_name,
            select=select,
            keys=keys,
            features=features,
            transform=transform,
        )
        train_dataloader = DataLoader(train_data, **self.loader_kwargs)
        valid_dataloader = DataLoader(valid_data, **self.loader_kwargs)

        parameters = inspect.signature(self.compute_metrics).parameters
        compute_metrics_kwargs = {
            k: getattr(self, k) for k in parameters if hasattr(self, k)
        }

        validation_sets = {
            vs: self.get_validation_dataset(database_type, database_name, vs)
            for vs in validation_sets
        }
        metrics = [database.load('metric', m) for m in metrics]

        return Trainer(
            models,
            model_names,
            self.objective,
            optimizers,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            compute_metrics=lambda validation_set: self.compute_metrics(
                validation_set,
                models=models,
                keys=keys,
                metrics=metrics,
                predict_kwargs=self.loader_kwargs,
                **compute_metrics_kwargs,
            ),
            save=lambda x, y: self.save_models(database, x, y),
            stopping_criterion=lambda x, y: self.stopping_criterion(
                x,
                y,
                watch=self.watch,
                max_iterations=self.max_iterations,
                no_improve_then_stop=self.no_improve_then_stop,
            ),
            saving_criterion=lambda x: self.saving_criterion(x, watch=self.watch),
            validation_sets=validation_sets,
            validation_interval=self.validation_interval,
        )


class Trainer:
    def __init__(
        self,
        models,
        model_names,
        objective,
        optimizers,
        train_dataloader,
        valid_dataloader,
        compute_metrics,
        save,
        saving_criterion,
        stopping_criterion,
        validation_interval,
        validation_sets,
    ):
        self.models = models
        self.model_names = model_names
        self.objective = objective
        self.optimizers = optimizers
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.compute_metrics = compute_metrics
        self.save = save
        self.saving_criterion = saving_criterion
        self.stopping_criterion = stopping_criterion
        self.validation_sets = validation_sets
        self.validation_interval = validation_interval

        self.metrics = ExtensibleDict(lambda: [])

    def log(self, **kwargs):
        out = ''
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out += f'{k}/{kk}: {vv}; '
            else:
                out += f'{k}: {v}; '
        logging.info(out)

    @staticmethod
    def apply_models_to_batch(batch, models):
        batch = to_device(batch, device_of(models[0]))
        output = []
        for subbatch, model in list(zip(batch, models)):
            if isinstance(model, torch.nn.Module) and hasattr(model, 'train_forward'):
                output.append(model.train_forward(subbatch))
            elif isinstance(model, torch.nn.Module):
                output.append(model(subbatch))
            else:
                output.append(subbatch)
        return output

    def take_step(self, batch):
        outputs = self.apply_models_to_batch(batch, self.models)
        objective_value = self.objective(*outputs)
        for opt in self.optimizers:
            opt.zero_grad()
        objective_value.backward()
        for opt in self.optimizers:
            opt.step()
        return objective_value

    def compute_validation_objective(self):
        objective_values = []
        for batch in self.valid_dataloader:
            outputs = self.apply_models_to_batch(batch, self.models)
            objective_value = self.objective(*outputs)
            objective_values.append(objective_value)
        return sum(objective_values) / len(objective_values)

    def __call__(self):
        iteration = 0
        while True:
            for batch in self.train_dataloader:
                train_objective = self.take_step(batch)
                self.log(fold='TRAIN', iteration=iteration, objective=train_objective)
                if iteration % self.validation_interval == 0:
                    valid_loss = self.compute_validation_objective()
                    all_metrics = {}
                    for vs in self.validation_sets:
                        metrics = self.compute_metrics(self.validation_sets[vs])
                        metrics = {f'{vs}/{k}': metrics[k] for k in metrics}
                        all_metrics.update(metrics)
                    all_metrics.update({'objective': valid_loss})
                    self.metrics.append(all_metrics)
                    self.log(fold='VALID', iteration=iteration, **all_metrics)
                    if self.saving_criterion(self.metrics):
                        self.save(self.model_names, self.models)
                    stop = self.stopping_criterion(self.metrics, iteration)
                    if stop:
                        return
                iteration += 1
