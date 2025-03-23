import dataclasses as dc
import typing as t

import torch
from superduper import logging
from superduper.base.datalayer import Datalayer
from superduper.base.query_dataset import QueryDataset
from superduper.components.dataset import Dataset
from superduper.components.model import Trainer
from torch.utils.data import DataLoader

from superduper_torch.model import BasicDataset, TorchModel


class TorchTrainer(Trainer):
    """
    Configuration for the PyTorch trainer.

    :param objective: Objective function
    :param loader_kwargs: Kwargs for the dataloader
    :param max_iterations: Maximum number of iterations
    :param no_improve_then_stop: Number of iterations to wait for improvement
                                 before stopping
    :param download: Whether to download the data
    :param validation_interval: How often to validate
    :param listen: Which metric to listen to for early stopping
    :param optimizer_cls: Optimizer class
    :param optimizer_kwargs: Kwargs for the optimizer
    :param optimizer_state: Latest state of the optimizer for contined training
    :param collate_fn: Collate function for the dataloader
    :param metric_values: Metric values
    """

    objective: t.Callable
    loader_kwargs: t.Dict = dc.field(default_factory=dict)
    max_iterations: int = 10**100
    no_improve_then_stop: int = 5
    download: bool = False
    validation_interval: int = 100
    listen: str = 'objective'
    optimizer_cls: str = 'Adam'
    optimizer_kwargs: t.Dict = dc.field(default_factory=dict)
    optimizer_state: t.Optional[t.Dict] = None
    collate_fn: t.Optional[t.Callable] = None
    metric_values: t.Dict = dc.field(default_factory=dict)

    def get_optimizers(self, model):
        """Get the optimizers for the model.

        :param model: Model
        """
        cls_ = getattr(torch.optim, self.optimizer_cls)
        optimizer = cls_(model.parameters(), **self.optimizer_kwargs)
        if self.optimizer_state is not None:
            self.optimizer.load_state_dict(self.optimizer_state)
            self.optimizer_state = None
        return (optimizer,)

    def _create_loader(self, dataset):
        dataset = BasicDataset(
            dataset, transform=self.transform, signature=self.signature
        )
        return torch.utils.data.DataLoader(
            dataset,
            **self.loader_kwargs,
            collate_fn=self.collate_fn,
        )

    def fit(
        self,
        model: TorchModel,
        db: Datalayer,
        train_dataset: QueryDataset,
        valid_dataset: QueryDataset,
    ):
        """Fit the model.

        :param model: Model
        :param db: Datalayer
        :param train_dataset: Training dataset
        :param valid_dataset: Validation dataset
        """
        train_dataloader = self._create_loader(train_dataset)
        valid_dataloader = self._create_loader(valid_dataset)
        return self._fit_with_dataloaders(
            model,
            db,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            validation_sets=model.validation.datasets if model.validation else None,
        )

    def take_step(self, model, batch, optimizers):
        """Take a step in the optimization.

        :param model: Model
        :param batch: Batch of data
        :param optimizers: Optimizers
        """
        if self.signature == '*args':
            outputs = model.train_forward(*batch)
        elif self.signature == 'singleton':
            outputs = model.train_forward(batch)
        elif self.signature == '**kwargs':
            outputs = model.train_forward(**batch)
        elif self.signature == '*args,**kwargs':
            outputs = model.train_forward(*batch[0], **batch[1])
        objective_value = self.objective(*outputs)
        for opt in optimizers:
            opt.zero_grad()
        objective_value.backward()
        for opt in optimizers:
            opt.step()
        return objective_value

    def compute_validation_objective(self, model, valid_dataloader):
        """Compute the validation objective.

        :param model: Model
        :param valid_dataloader: Validation dataloader to use
        """
        objective_values = []
        with model.evaluating(), torch.no_grad():
            for batch in valid_dataloader:
                objective_values.append(
                    self.objective(*model.train_forward(*batch)).item()
                )
            return sum(objective_values) / len(objective_values)

    def _fit_with_dataloaders(
        self,
        model,
        db: Datalayer,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        validation_sets: t.Optional[t.Sequence[Dataset]] = None,
    ):
        if validation_sets is None:
            validation_sets = []

        model.train()
        iteration = 0

        optimizers = self.get_optimizers(model)

        while True:
            for batch in train_dataloader:
                train_objective = self.take_step(model, batch, optimizers)
                self.log(fold='TRAIN', iteration=iteration, objective=train_objective)

                if iteration % self.validation_interval == 0:
                    valid_loss = self.compute_validation_objective(
                        model, valid_dataloader
                    )
                    all_metrics = {}
                    for vs in validation_sets:
                        m = model.validate(
                            key=self.key, dataset=vs, metrics=model.validation.metrics
                        )
                        all_metrics.update(
                            {f'{vs.identifier}/{k}': v for k, v in m.items()}
                        )
                    all_metrics.update({'objective': valid_loss})
                    self.append_metrics(all_metrics)
                    self.log(fold='VALID', iteration=iteration, **all_metrics)
                    if self.saving_criterion():
                        db.apply(model, force=True, jobs=False)
                    stop = self.stopping_criterion(iteration)
                    if stop:
                        return
                iteration += 1

    def append_metrics(self, d: t.Dict[str, float]) -> None:
        """Append metrics to the metric_values dict.

        :param d: Metrics to append
        """
        if self.metric_values is not None:
            for k, v in d.items():
                self.metric_values.setdefault(k, []).append(v)

    def stopping_criterion(self, iteration):
        """Check if the training should stop.

        :param iteration: Current iteration
        """
        max_iterations = self.max_iterations
        no_improve_then_stop = self.no_improve_then_stop
        if isinstance(max_iterations, int) and iteration >= max_iterations:
            return True
        if isinstance(no_improve_then_stop, int):
            if self.listen == 'objective':
                to_listen = [-x for x in self.metric_values['objective']]
            else:
                to_listen = self.metric_values[self.listen]
            if max(to_listen[-no_improve_then_stop:]) < max(to_listen):
                logging.info('early stopping triggered!')
                return True
        return False

    def saving_criterion(self):
        """Check if the model should be saved."""
        if self.listen == 'objective':
            to_listen = [-x for x in self.metric_values['objective']]
        else:
            to_listen = self.metric_values[self.listen]
        if all([to_listen[-1] >= x for x in to_listen[:-1]]):
            return True
        return False

    def log(self, **kwargs):
        """Log the training progress.

        :param kwargs: Key-value pairs to log
        """
        out = ''
        for k, v in kwargs.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    out += f'{k}/{kk}: {vv}; '
            else:
                out += f'{k}: {v}; '
        logging.info(out)
