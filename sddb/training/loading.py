from torch.utils import data
import pymongo


class BasicDataset(data.Dataset):
    def __init__(self, documents, transform=None):
        super().__init__()
        self.documents = documents
        self.transform = transform

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, item):
        if self.transform is None:
            return self.documents[item]
        else:
            r = self.transform(self.documents[item])
            return r


class MongoIterable(data.IterableDataset):
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
    def __init__(self, client, database, collection, filter=None, transform=None, download=False,
                 splitter=None):
        super().__init__()
        self._client = client
        self._database = database
        self._collection = collection
        self.splitter = splitter
        self.transform = transform
        self.filter = filter if filter is not None else {}
        self.download = download

        from sddb.utils import Progress
        if not self.download:
            cursor = self.collection.find(self.filter, {'_id': 1})
            self.ids = []
            docs = Progress()(cursor, total=len(self))
            docs.set_description(f'downloading ids for {filter}')
            for r in docs:
                self.ids.append(r['_id'])
        else:
            cursor = self.collection.find(self.filter)
            self.ids = []
            self.documents = {}
            docs = Progress()(cursor, total=len(self))
            docs.set_description(f'downloading records for {filter}')
            for r in docs:
                self.ids.append(r['_id'])
                self.documents[r['_id']] = r

    @property
    def client(self):
        from sddb.client import SddbClient
        return SddbClient(**self._client)

    @property
    def database(self):
        return self.client[self._database]

    @property
    def collection(self):
        return self.database[self._collection]

    def __len__(self):
        return self.collection.count_documents(self.filter)

    def __getitem__(self, item):
        if self.download:
            r = self.documents[self.ids[item]]
        else:
            r = self.collection.find_one({'_id': self.ids[item]})

        if self.splitter is not None:
            r = self.splitter(r)

        if '_id' in r:
            del r['_id']

        if self.transform is not None:
            return self.transform(r)
        else:
            return r
