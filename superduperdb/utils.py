import os
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
import requests
import signal
import sys
import warnings

import torch
import torch.utils.data


class MongoStyleDict(dict):
    """
    Dictionary object mirroring how fields can be referred to and set in MongoDB.

    >>> d = MongoStyleDict({'a': {'b': 1}})
    >>> d['a.b']
    1

    Set deep fields directly with string keys:
    >>> d['a.c'] = 2
    >>> d
    {'a': {'b': 1, 'c': 2}}

    Parent keys should exist in order to set subfields:
    >>> d['a.d.e'] = 3
    Traceback (most recent call last):
    ...
    KeyError: 'd'
    """
    def __getitem__(self, item):
        if '.' not in item:
            return super().__getitem__(item)
        parts = item.split('.')
        parent = parts[0]
        child = '.'.join(parts[1:])
        sub = MongoStyleDict(self.__getitem__(parent))
        return sub[child]

    def __setitem__(self, key, value):
        if '.' not in key:
            super().__setitem__(key, value)
        else:
            parent = key.split('.')[0]
            child = '.'.join(key.split('.')[1:])
            parent_item = MongoStyleDict(self[parent])
            parent_item[child] = value
            self[parent] = parent_item


def create_batch(args):
    """
    Create a singleton batch in a manner similar to the PyTorch dataloader

    :param args: single data point for batching

    >>> create_batch(3.).shape
    torch.Size([1])
    >>> x, y = create_batch([torch.randn(5), torch.randn(3, 7)])
    >>> x.shape
    torch.Size([1, 5])
    >>> y.shape
    torch.Size([1, 3, 7])
    >>> d = create_batch(({'a': torch.randn(4)}))
    >>> d['a'].shape
    torch.Size([1, 4])
    """
    if isinstance(args, (tuple, list)):
        return tuple([create_batch(x) for x in args])
    if isinstance(args, dict):
        return {k: create_batch(args[k]) for k in args}
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, (float, int)):
        return torch.tensor([args])
    raise TypeError('only tensors and tuples of tensors recursively supported...')  # pragma: no cover


def unpack_batch(args):
    """
    Unpack a batch into lines of tensor output.

    :param args: a batch of model outputs

    >>> unpack_batch(torch.randn(1, 10))[0].shape
    torch.Size([10])
    >>> out = unpack_batch([torch.randn(2, 10), torch.randn(2, 3, 5)])
    >>> type(out)
    <class 'list'>
    >>> len(out)
    2
    >>> out = unpack_batch({'a': torch.randn(2, 10), 'b': torch.randn(2, 3, 5)})
    >>> [type(x) for x in out]
    [<class 'dict'>, <class 'dict'>]
    >>> out[0]['a'].shape
    torch.Size([10])
    >>> out[0]['b'].shape
    torch.Size([3, 5])
    >>> out = unpack_batch({'a': {'b': torch.randn(2, 10)}})
    >>> out[0]['a']['b'].shape
    torch.Size([10])
    >>> out[1]['a']['b'].shape
    torch.Size([10])
    """

    if isinstance(args, torch.Tensor):
        return [args[i] for i in range(args.shape[0])]
    else:
        if isinstance(args, list) or isinstance(args, tuple):
            tmp = [unpack_batch(x) for x in args]
            batch_size = len(tmp[0])
            return [[x[i] for x in tmp] for i in range(batch_size)]
        elif isinstance(args, dict):
            tmp = {k: unpack_batch(v) for k, v in args.items()}
            batch_size = len(next(iter(tmp.values())))
            return [{k: v[i] for k, v in tmp.items()} for i in range(batch_size)]
        else: # pragma: no cover
            raise NotImplementedError


class TimeoutException(Exception):
    ...


def timeout_handler(signum, frame):  # pragma: no cover
    raise TimeoutException()


@contextmanager
def timeout(seconds):  # pragma: no cover
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class Downloader:
    def __init__(
        self,
        urls,
        collection,
        ids=None,
        keys=None,
        n_workers=20,
        raises=True,
        max_queue=10,
        headers=None,
        skip_existing=True,
        timeout=None,
        update_db=True,
    ):
        """
        Parallel file downloader
        :param urls: list of file urls
        :param files: list of destination paths for saving
        :param n_workers: number of workers over which to parallelize
        :param n_callback_workers: number of workers over which to parallelize callbacks
        :param max_queue: Maximum number of tasks in the multiprocessing queue.
        """
        self.urls = urls
        self.ids = ids
        self.keys = keys
        self.collection = collection
        self.n_workers = n_workers
        self.raises = raises
        self.max_queue = max_queue
        self.failed = 0
        self.headers = headers
        self.skip_existing = skip_existing
        self.timeout = timeout
        self.update_db = update_db
        if not self.update_db:  # pragma: no cover
            self.results = {}

        assert len(ids) == len(urls)

    def go(self):
        """
        Download all files
        Uses a :py:class:`multiprocessing.pool.ThreadPool` to parallelize
                          connections.
        :param test: If *True* perform a test run.
        """
        print(f'number of workers {self.n_workers}')
        prog = progressbar(total=len(self.urls))
        prog.prefix = 'downloading from urls'
        self.failed = 0
        prog.prefx = "failed: 0"
        request_session = requests.Session()
        request_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=self.n_workers if self.n_workers else 1,
            pool_maxsize=self.n_workers * 10
        )
        request_session.mount("http://", request_adapter)
        request_session.mount("https://", request_adapter)

        def f(i):
            prog.update()
            try:
                if self.timeout is not None:  # pragma: no cover
                    with timeout(self.timeout):
                        self.download(i, request_session=request_session)
                else:
                    self.download(i, request_session=request_session)
            except TimeoutException:  # pragma: no cover
                print(f'timed out {i}')
            except KeyboardInterrupt:  # pragma: no cover
                raise
            except Exception as e:  # pragma: no cover
                if self.raises:
                    raise e
                warnings.warn(str(e))
                self.failed += 1
                prog.prefix = f"failed: {self.failed} [{e}]"

        if self.n_workers == 0:
            self._sequential_go(f)
            return

        self._parallel_go(f)

    def _parallel_go(self, f):
        pool = ThreadPool(self.n_workers)
        try:
            pool.map(f, range(len(self.urls)))
        except KeyboardInterrupt:  # pragma: no cover
            print("--keyboard interrupt--")
            pool.terminate()
            pool.join()
            sys.exit(1)

        pool.close()
        pool.join()

    def _sequential_go(self, f):
        for i in range(len(self.urls)):
            f(i)

    def download(self, i, request_session):
        """
        Download i-th url file
        """
        url = self.urls[i]
        _id = self.ids[i]

        if url.startswith('http'):
            r = request_session.get(url, headers=self.headers)
        elif url.startswith('file'):
            with open(url.split('file://')[-1], 'rb') as f:
                r = lambda: None
                r.content = f.read()
                r.status_code = 200
        else:
            raise NotImplementedError('unknown URL type...')

        if r.status_code != 200:  # pragma: no cover
            raise Exception(f"Non-200 response. ({r.status_code})")

        if self.update_db:
            self.collection.update_one({'_id': self.ids[i]},
                                       {'$set': {f'{self.keys[i]}._content.bytes': r.content}},
                                       refresh=False)
        else:  # pragma: no cover
            self.results[self.ids[i]] = r.content


class _UpdateableProgress:
    def __init__(self, total, size=None, prefix='', file=sys.stdout):
        self.total = total
        if size is None:
            try:
                size = os.get_terminal_size().columns
            except OSError as e:
                size = 80
        self.size = size
        self.file = file
        self.prefix = prefix
        self.j = 0

    def update(self, it=1):
        self.j += it
        ending = f'{self.j}/{self.total}'
        current_size = self.size - len(ending) - 4 - len(self.prefix)
        x = int(current_size * self.j / self.total)
        beginning = f'{self.prefix} [{"#" * x}{"." * (current_size - x)}] '
        print(beginning + ending, end='\r', file=self.file, flush=True)


def progressbar(*args, **kwargs):
    """
    """
    if args and args[0] is not None:
        return _progressbar(*args, **kwargs)
    else:
        return _UpdateableProgress(*args, **kwargs)


def _progressbar(it=None, prefix="", size=None, out=sys.stdout, total=None):
    if size is None:
        try:
            size = os.get_terminal_size().columns
        except OSError as e:
            size = 80
    if total is None:
        total = len(it)

    def show(j, end='\r'):
        ending = f'{j}/{total}'
        current_size = size - len(ending) - 4 - len(prefix)
        x = int(current_size * j / total)
        beginning = f'{prefix} [{"#" * x}{"." * (current_size - x)}] '
        print(beginning + ending, end=end, file=out, flush=True)

    show(0)
    for i, item in enumerate(it):
        yield item
        if i + 1 == total:
            show(i + 1, end=None)
        else:
            show(i + 1)


class ArgumentDefaultDict(defaultdict):
    def __getitem__(self, item):
        if item not in self.keys():
            self[item] = self.default_factory(item)
        return super().__getitem__(item)