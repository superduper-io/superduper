import pymongo
from torch.utils import data


def load(file_id, filesystem):
    """
    :param file_id: id in file system
    :param filesystem: MongoDB gridfs filesystem
    """
    return filesystem.get(file_id).read()


def save(bytes_, filesystem):
    """
    :param bytes_: content to save
    :param filesystem: MongoDB gridfs filesystem
    """
    return filesystem.put(bytes_)


class MongoIterable(data.IterableDataset):  # pragma: no cover
    """
    Dataset iterating over a query without needing to download the whole thing first.
    """
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