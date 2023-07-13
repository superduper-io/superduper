import random
import time
from collections import defaultdict
import torch.utils.data

from sddb import client
from sddb.training.loading import QueryDataset
from sddb.utils import MongoStyleDict, apply_model


class Trainer:
    def __init__(self,
                 train_name,
                 client,
                 database,
                 collection,
                 encoders,
                 fields,
                 save_names,
                 metrics=None,
                 splitter=None,
                 loss=None,
                 batch_size=100,
                 optimizers=(),
                 lr=0.0001,
                 num_workers=0,
                 projection=None,
                 filter=None,
                 features=None,
                 n_epochs=100,
                 save=None,
                 watch='loss',
                 download=False):

        self.train_name = train_name
        self._client = client
        self._database = database
        self._collection = collection
        self._collection_object = None

        self.encoders = encoders
        self.fields = fields
        self.splitter = splitter
        self.loss = loss
        self.features = features
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.filter = filter if filter is not None else {}
        self.n_epochs = n_epochs
        self.download = download
        self.training_id = str(int(time.time() * 100))
        self.train_data = QueryDataset(
            client=self._client,
            database=self._database,
            collection=self._collection,
            filter={**self.filter, '_fold': 'train'},
            transform=self.apply_splitter_and_encoders,
            projection=projection,
            features=features,
        )
        self.valid_data = QueryDataset(
            client=self._client,
            database=self._database,
            collection=self._collection,
            filter={**self.filter, '_fold': 'valid'},
            download=True,
            transform=self.apply_splitter_and_encoders,
            projection=projection,
            features=features,
        )
        self.best = []
        self.metrics = metrics if metrics is not None else {}
        if isinstance(self.fields, str):
            self.fields = (self.fields,)
        if not isinstance(self.encoders, tuple) and not isinstance(self.encoders, list):
            self.encoders = (self.encoders,)
        self.learn_fields = self.fields
        self.learn_encoders = self.encoders
        if len(self.fields) == 1:
            self.learn_fields = (self.fields[0], self.fields[0])
            self.learn_encoders = (self.encoders[0], self.encoders[0])
        self.optimizers = optimizers if optimizers else [
            self._get_optimizer(encoder, lr) for encoder in self.encoders
            if isinstance(encoder, torch.nn.Module)
        ]
        self.save_names = save_names
        self.watch = watch
        self.metric_values = defaultdict(lambda: [])
        self.lr = lr
        self.weights_dict = defaultdict(lambda: [])
        self._weights_choices = {}
        self._init_gradients()
        self.save = save

    def _init_gradients(self):
        for i, e in enumerate(self.encoders):
            try:
                sd = e.state_dict()
            except AttributeError:
                continue
            self._weights_choices[i] = {}
            for p in sd:
                if len(sd[p].shape) == 2:
                    indexes = [(random.randrange(sd[p].shape[0]), random.randrange(sd[p].shape[1]))
                                for _ in range(min(10, max(sd[p].shape[0], sd[p].shape[1])))]
                else:
                    assert len(sd[p].shape) == 1
                    indexes = [(random.randrange(sd[p].shape[0]),)
                               for _ in range(min(10, sd[p].shape[0]))]
                self._weights_choices[i][p] = indexes

    def log_gradients(self):
        for i, f in enumerate(self._weights_choices):
            sd = self.encoders[i].state_dict()
            for p in self._weights_choices[f]:
                indexes = self._weights_choices[f][p]
                tmp = []
                for ind in indexes:
                    param = sd[p]
                    if len(ind) == 1:
                        tmp.append(param[ind[0]].item())
                    elif len(ind) == 2:
                        tmp.append(param[ind[0], ind[1]].item())
                    else:
                        raise Exception('3d tensors not supported')
                self.weights_dict[p].append(tmp)

    def _save_metrics(self):
        self.collection[self.sub_collection].update_one(
            {'name': self.train_name},
            {'$set': {'metric_values': self.metric_values, 'weights': self.weights_dict}}
        )

    def _save_best_model(self):
        agg = min if self.watch == 'loss' else max
        if self.watch == 'loss' and self.metric_values['loss'][-1] == agg(self.metric_values['loss']):
            for sn, encoder in zip(self.save_names, self.encoders):
                self.save(sn, encoder)

    def _load_metrics(self):
        for m in self.metrics:
            self.metrics[m] = self.collection.load_metric(m)

    def calibrate(self, i):
        if not hasattr(self.encoders[i], 'calibrate'):
            return

        if self.fields[i] != '_base':
            data_points = []
            for r in self.collection.find(self.filter, {self.fields[i]: 1}):
                data_points.append(r[self.fields[i]])
            self.encoders[i].calibrate(data_points)

    @property
    def client(self):
        return client.SddbClient(**self._client)

    @property
    def database(self):
        return self.client[self._database]

    @property
    def collection(self):
        if self._collection_object is None:
            self._collection_object = self.database[self._collection]
        return self._collection_object

    def _get_optimizer(self, encoder, lr):
        learnable_parameters = [x for x in encoder.parameters() if x.requires_grad]
        return torch.optim.Adam(learnable_parameters, lr=lr)

    def apply_splitter_and_encoders(self, sample):
        if self.splitter is not None:
            sample = self.splitter(sample)
        else:
            # This is for example the classification case
            sample = [sample for _ in self.learn_fields]
        return _Mapped([x.preprocess for x in self.learn_encoders], self.learn_fields)(sample)

    @property
    def data_loaders(self):
        return (
            torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size,
                                        num_workers=self.num_workers),
            torch.utils.data.DataLoader(self.valid_data, batch_size=self.batch_size,
                                        num_workers=self.num_workers),
        )

    def log_progress(self, **kwargs):
        out = ''
        for k, v in kwargs.items():
            out += f'{k}: {v}; '
        print(out)

    @staticmethod
    def apply_models_to_batch(batch, models):
        output = []
        for subbatch, model in list(zip(batch, models)):
            output.append(model.forward(subbatch))
        return output

    def take_step(self, loss):
        for opt in self.optimizers:
            opt.zero_grad()
        loss.backward()
        for opt in self.optimizers:
            opt.step()
        self.log_gradients()
        return loss


class ImputationTrainer(Trainer):

    sub_collection = '_imputations'

    def __init__(self, *args, inference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_model = inference_model if inference_model is not None else self.encoders[0]

    def validate_model(self, data_loader, *args, **kwargs):
        for e in self.encoders:
            if hasattr(e, 'eval'):
                e.eval()
        docs = list(self.collection.find(
            {**self.filter, '_fold': 'valid'},
            features=self.features,
        ))
        inputs_ = [r[self.fields[0]] for r in docs]
        targets = [r[self.fields[1]] for r in docs]
        outputs = apply_model(
            self.inference_model,
            inputs_,
            single=False,
            verbose=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        for metric in self.metrics:
            self.metric_values[metric].append(self.metrics[metric](outputs, targets))
        loss = []
        for batch in data_loader:
            outputs = self.apply_models_to_batch(batch, self.encoders)
            loss.append(self.loss(*outputs).item())
        self.metric_values['loss'].append(sum(loss) / len(loss))
        self._save_metrics()
        for e in self.encoders:
            if hasattr(e, 'train'):
                e.train()

    def train(self):
        for i in range(len(self.encoders)):
            self.calibrate(i)
        for e in self.encoders:
            if hasattr(e, 'train'):
                e.train()

        it = 0
        epoch = 0
        while True:
            train_loader, valid_loader = self.data_loaders
            for batch in train_loader:
                outputs = self.apply_models_to_batch(batch, self.encoders)
                l_ = self.take_step(self.loss(*outputs))
                self.log_progress(fold='TRAIN', iteration=it, loss=l_.item())
                it += 1

            self.validate_model(valid_loader, epoch)
            self._save_best_model()
            self.log_progress(
                fold='VALID',
                iteration=it,
                **{k: v[-1] for k, v in self.metric_values.items()},
            )

            epoch += 1
            if self.n_epochs is not None and epoch >= self.n_epochs:
                return


class _Mapped:
    def __init__(self, functions, keys):
        self.functions = functions
        self.keys = keys

    def __call__(self, args):
        args = [MongoStyleDict(r) for r in args]
        inputs = [r[k] for r, k in zip(args, self.keys)]
        return [f(i) for i, f in zip(inputs, self.functions)]


class RepresentationTrainer(Trainer):
    sub_collection = '_semantic_indexes'

    def __init__(self, semantic_index_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_index_name = semantic_index_name

    def prepare_validation_set(self):
        for r in self.collection.find({**self.filter, '_fold': 'valid'}):
            left, right = self.splitter(r)
            self.collection.replace_one(
                {'_id': r['_id']},
                {'_backup': r, '_query': left, **right},
                refresh=False,
            )

    def validate_model(self, data_loader, epoch):
        active_models = self.collection.active_models

        self.collection.semantic_index = self.semantic_index_name
        self.collection.active_models = self.collection.semantic_index['active_models']
        self.collection.update_many({'_fold': 'valid'}, {'$set': {'epoch': epoch}})
        self.collection.active_models = active_models

        metrics_values = defaultdict(lambda: [])

        for r in self.collection.find({**self.filter, '_fold': 'valid'}):
            results = list(self.collection.find(
                {'$like': {'document': r, 'n': 100}},
                {'_id': 1}
            ))
            for metric in self.metrics:
                metrics_values[metric].append(self.metrics[metric](r, results))

        for batch in data_loader:
            metrics_values['loss'].append(self.loss(*batch))

        for k in metrics_values:
            metrics_values[k] = sum(metrics_values[k]) / len(metrics_values[k])

        return metrics_values

    def train(self):

        for i in range(len(self.encoders)):
            self.calibrate(i)

        self.prepare_validation_set()

        it = 0
        epoch = 0

        while True:

            train_loader, valid_loader = self.data_loaders

            for batch in train_loader:
                outputs = self.apply_models_to_batch(batch, self.learn_encoders)
                l_ = self.take_step(self.loss(*outputs))
                self.log_progress(fold='TRAIN', iteration=it, loss=l_.item())
                it += 1

            metrics = self.validate_model(valid_loader, epoch)
            self.log_progress(fold='VALID', iteration=it, **metrics)

            epoch += 1
            if self.n_epochs is not None and epoch >= self.n_epochs:
                return
