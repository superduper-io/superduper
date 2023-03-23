import importlib
from torch.utils import data
import pymongo

from superduperdb import cf
from superduperdb.utils import get_database_from_database_type, MongoStyleDict


class MongoIterable(data.IterableDataset):  # pragma: no cover
    def __init__(self, client, database, collection, transform=None, filter=None):
        super().__init__()
        self._client = client
        self._database = database
        self._collection = collection
        self.transform = transform
        self.filter = filter if filter is not None else {}

    def __len__(self):
        return self.collection.count_documents(self.filter)

    @property
    def client(self):
        return pymongo.MongoClient(**self._client)

    @property
    def database(self):
        return self.client[self._database]

    @property
    def collection(self):
        return self.database[self._collection]

    def __iter__(self):
        worker_info = data.get_worker_info()
        if worker_info is None:
            for r in self.collection.find(self.filter, {'_id': 0}):
                if self.transform is not None:
                    yield self.transform(r)
                else:
                    yield r
        else:
            n_documents = self.collection.count_documents(self.filter)
            per_worker = (
                n_documents // worker_info.num_workers
                if worker_info.id < worker_info.num_workers - 1
                else (
                        len(self) % n_documents // worker_info.num_workers
                        + n_documents // worker_info.num_workers
                )
            )
            skip = worker_info.id * per_worker
            for r in self.collection.find(self.filter, {'_id': 0}).skip(skip).limit(per_worker):
                if self.transform is not None:
                    yield self.transform(r)
                else:
                    yield r


class QueryDataset(data.Dataset):
    def __init__(self, database_type, database, query_params, fold='train', suppress=(), transform=None):
        super().__init__()

        self._database = None
        self._database_type = database_type
        self._database_name = database

        self.transform = transform if transform else lambda x: x
        query_params = self.database._format_fold_to_query(query_params, fold)
        self._documents = list(self.database.execute_query(*query_params))
        self.suppress = suppress

    @property
    def database(self):
        if self._database is None:
            self._database = get_database_from_database_type(self._database_type, self._database_name)
        return self._database

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, item):
        r = MongoStyleDict(self._documents[item])
        for k in self.suppress:
            del r[k]
        return self.transform(self._documents[item])
