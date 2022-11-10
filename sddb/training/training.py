import time
from collections import defaultdict

import torch.utils.data

from sddb.client import SddbClient
from sddb.training.loading import QueryDataset
from sddb.utils import MongoStyleDict, apply_model


class Trainer:
    def __init__(self,
                 client,
                 database,
                 collection,
                 encoders,
                 fields,
                 metrics=None,
                 splitter=None,
                 loss=None,
                 batch_size=100,
                 optimizers=(),
                 lr=0.0001,
                 num_workers=0,
                 filter=None,
                 n_epochs=None,
                 download=False):

        self._client = client
        self._database = database
        self._collection = collection
        self._collection_object = None
        self.encoders = encoders
        self.fields = fields
        self.splitter = splitter
        self.loss = loss
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
            transform=lambda x: self.apply_splitter_and_encoders(x),
        )
        self.valid_data = QueryDataset(
            client=self._client,
            database=self._database,
            collection=self._collection,
            filter={**self.filter, '_fold': 'valid'},
            download=True,
            transform=lambda x: self.apply_splitter_and_encoders(x),
        )
        self.metrics = metrics if metrics is not None else ()
        if isinstance(self.fields, str):
            self.fields = (self.fields,)
        if not isinstance(self.encoders, tuple):
            self.encoders = (self.encoders,)
        self.learn_fields = self.fields
        self.learn_encoders = self.encoders
        if len(self.fields) == 1:
            self.learn_fields = (self.fields[0], self.fields[0])
            self.learn_encoders = (self.encoders[0], self.encoders[0])
        self.optimizers = optimizers if optimizers else (
            self._get_optimizer(encoder, lr) for encoder in self.encoders
        )

    def load_metrics(self):
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
        return SddbClient(**self._client)

    @property
    def database(self):
        return self.client[self._database]

    @property
    def collection(self):
        if self._collection_object is None:
            self._collection_object = self.database[self._collection]
        return self._collection_object

    def _get_optimizer(self, encoder, lr):
        return torch.optim.Adam(encoder.parameters(), lr=lr)

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
        for subbatch, model in zip(batch, models):
            output.append(model.forward(subbatch))
        return output

    def take_step(self, loss):
        for opt in self.optimizers:
            opt.zero_grad()

        loss.backward()

        for opt in self.optimizers:
            opt.step()

        return loss


class ImputationTrainer(Trainer):

    def __init__(self, *args, inference_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_model = inference_model if inference_model is not None else self.encoders[0]

    def validate_model(self, data_loader, *args, **kwargs):

        metrics = defaultdict(lambda: [])
        docs = list(self.collection.find({**self.filter, '_fold': 'valid'}))
        outputs = apply_model(
            self.inference_model,
            [r[self.fields[0]] for r in docs],
            single=False,
            verbose=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

        for output in outputs:
            for metric in self.metrics:
                metrics[metric].append(self.metrics[metric](*output))

        for batch in data_loader:
            metrics['loss'].append(self.loss(*batch))

        for k in metrics:
            metrics[k] = sum(metrics[k]) / len(metrics[k])

        return metrics

    def train(self):
        for i in range(len(self.encoders)):
            self.calibrate(i)

        name = f'_training_index_{self.training_id}'
        self.collection.create_imputation({
            'name': name,
            'input': 'test',
            'label': 'fruit',
            'models': {
                'test': 'dummy',
                'type': 'in_memory',
                'object': self.encoders[0],
                'active': False,
            }
        })

        it = 0
        epoch = 0

        while True:

            train_loader, valid_loader = self.data_loaders

            for batch in train_loader:
                outputs = self.apply_models_to_batch(batch, self.encoders)
                l_ = self.take_step(self.loss(*outputs))
                self.log_progress(fold='TRAIN', iteration=it, loss=l_.item())
                it += 1

            metrics = self.validate_model(valid_loader, epoch)
            self.log_progress(fold='VALID', iteration=it, **metrics)

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

    def prepare_validation_set(self):
        for r in self.collection.find({**self.filter, '_fold': 'valid'}):
            left, right = self.splitter(r)
            self.collection.replace_one(
                {'_id': r['_id']},
                {'_backup': r, '_query': left, **right},
            )

    def validate_model(self, data_loader, epoch):
        active_models = self.collection.active_models
        self.collection.active_models = \
            [f'_training_index_{self.training_id}:{len(self.fields) - 1}']
        # use update to force hashes to be updated
        try:
            del self.collection._all_hash_sets[self.collection.semantic_index_name]
        except KeyError:
            pass
        self.collection.update_many({'_fold': 'valid'}, {'$set': {'epoch': epoch}})
        self.collection.active_models = active_models
        metrics_values = defaultdict(lambda: [])

        for r in self.collection.find({**self.filter, '_fold': 'valid'}):
            results = self.collection.find(
                {'_fold': 'valid', self.fields[0]: {'$like': {'document': r['_query'], 'n': 100}}},
                {'_id': 1},
            )
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
        name = f'_training_index_{self.training_id}'
        self.collection.create_semantic_index({
            'name': name,
            'keys': self.fields,
            'models': [
                {
                    'name': f'{name}:{i}',
                    'object': self.encoders[i],
                    'active': i == len(self.fields) - 1,
                    'type': 'in_memory',
                    'args': {},
                    'filter': {'_fold': 'valid'},
                    'converter': 'sddb.models.converters.FloatTensor',
                    'key': self.fields[i],
                }
                for i, k in enumerate(self.fields)
            ],
        })
        self.collection.semantic_index = name

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
