from contextlib import contextmanager
import importlib
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import os
import requests
import signal
import sys
from time import sleep
import warnings

import torch
import torch.utils.data
import tqdm

from sddb.training.loading import BasicDataset


class MongoStyleDict(dict):
    def __getitem__(self, item):
        if '.' not in item:
            return super().__getitem__(item)
        else:
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
    """
    if isinstance(args, (tuple, list)):
        return tuple([create_batch(x) for x in args])
    if isinstance(args, dict):
        return {k: create_batch(args[k]) for k in args}
    if isinstance(args, torch.Tensor):
        return args.unsqueeze(0)
    if isinstance(args, (float, int)):
        return torch.tensor([args])
    raise TypeError('only tensors and tuples of tensors recursively supported...')


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
        else:
            raise NotImplementedError


def apply_model(model, args, single=True, verbose=False, **kwargs):
    """
    Apply model to args including pre-processing, forward pass and post-processing.

    :param model: model object including methods *preprocess*, *forward* and *postprocess*
    :param args: single or multiple data points over which to evaluate model
    :param single: toggle to apply model to single or multiple (batched) datapoints.
    :param verbose: display progress bar
    :param kwargs: key, value pairs to be passed to dataloader
    """
    if single:
        prepared = model.preprocess(args)
        singleton_batch = create_batch(prepared)
        output = model.forward(singleton_batch)
        output = unpack_batch(output)[0]
        if hasattr(model, 'postprocess'):
            return model.postprocess(output)
        return output
    else:
        inputs = BasicDataset(args, model.preprocess)
        loader = torch.utils.data.DataLoader(inputs, **kwargs)
        out = []
        if verbose:
            progress = Progress()(total=len(args))
        for batch in loader:
            tmp = model.forward(batch)
            tmp = unpack_batch(tmp)
            if hasattr(model, 'postprocess'):
                tmp = list(map(model.postprocess, tmp))
            out.extend(tmp)
            if verbose:
                progress.update(len(tmp))
        return out


def import_object(path):
    module = '.'.join(path.split('.')[:-1])
    object_ = path.split('.')[-1]
    module = importlib.import_module(module)
    return getattr(module, object_)


class TimeoutException(Exception):
    ...


def timeout_handler(signum, frame):
    raise TimeoutException()


@contextmanager
def timeout(seconds):
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
        files,
        n_workers=20,
        raises=True,
        callback=None,
        n_callback_workers=0,
        max_queue=10,
        headers=None,
        skip_existing=True,
        timeout=None,
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
        self.files = files
        self.n_workers = n_workers
        self.raises = raises
        self.callback = callback
        self.callback_pool = None
        self.n_callback_workers = n_callback_workers
        self.max_queue = max_queue
        self.failed = 0
        self.headers = headers
        self.skip_existing = skip_existing
        self.timeout = timeout

        assert len(files) == len(urls)

    def go(self):
        """
        Download all files
        Uses a :py:class:`multiprocessing.pool.ThreadPool` to parallelize
                          connections.
        :param test: If *True* perform a test run.
        """
        progress_bar = tqdm.tqdm(total=len(self.urls))
        progress_bar.set_description('downloading from urls')
        self.failed = 0
        progress_bar.set_description("failed: 0")
        print(f'number of workers {self.n_workers}')
        request_session = requests.Session()
        request_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=self.n_workers if self.n_workers else 1,
            pool_maxsize=self.n_workers * 10
        )
        request_session.mount("http://", request_adapter)
        request_session.mount("https://", request_adapter)

        def f(i):
            progress_bar.update()
            try:
                if self.timeout is not None:
                    with timeout(self.timeout):
                        self.download(i, request_session=request_session)
                else:
                    self.download(i, request_session=request_session)
            except TimeoutException:
                print(f'timed out {i}')
            except KeyboardInterrupt:  # pragma: no cover
                raise
            except Exception as e:
                if self.raises:  # pragma: no cover
                    raise e
                warnings.warn(str(e))
                self.failed += 1
                progress_bar.set_description(f"failed: {self.failed} [{e}]")

        if self.n_workers == 0:
            self._sequential_go(f)
            return

        self._parallel_go(f)

    def _parallel_go(self, f):
        if self.n_callback_workers > 0:
            self.callback_pool = Pool(self.n_callback_workers)
        else:
            self.callback_pool = None

        pool = ThreadPool(self.n_workers)
        try:
            pool.map(f, range(len(self.urls)))
        except KeyboardInterrupt:
            print("--keyboard interrupt--")
            pool.terminate()
            pool.join()
            if self.callback_pool is not None:
                self.callback_pool.terminate()
                self.callback_pool.join()
                self.callback_pool = None
            sys.exit(1)

        pool.close()
        pool.join()
        if self.callback_pool is not None:
            self.callback_pool.close()
            self.callback_pool.join()
            self.callback_pool = None

    def _sequential_go(self, f):
        for i in range(len(self.urls)):
            f(i)

    def _async_apply_callback(self, args):
        # this is to limit the size of the queue for the callback, without this, memory is filled quickly
        while True:
            if self.callback_pool._taskqueue.qsize() > self.max_queue:
                sleep(0.1)
            else:
                break
        self.callback_pool.apply_async(self.callback, args=args)

    def download(self, i, request_session=None):
        """
        Download i-th url file
        """
        url = self.urls[i]
        file_ = self.files[i]
        if self.skip_existing and os.path.isfile(file_):  # pragma: no cover
            return

        if "/" in file_:
            dir_ = "/".join(file_.split("/")[:-1])
            if not os.path.exists(dir_):
                os.system(f"mkdir -p {dir_}")

        if request_session is not None:
            r = request_session.get(url, headers=self.headers)
        else:
            r = requests.get(url, headers=self.headers)

        if r.status_code != 200:  # pragma: no cover
            raise Exception(f"Non-200 response. ({r.status_code})")

        if self.callback is None:  # pragma: no cover
            with open(file_, "wb") as f:
                f.write(r.content)
        else:
            if self.callback_pool is not None:
                self._async_apply_callback(args=(r.content, self.urls[i], self.files[i]))
            else:  # pragma: no cover
                self.callback(r.content, self.urls[i], self.files[i])


def basic_progress(iterator, *args, total=None, **kwargs):
    if total is None:
        try:
            total = len(iterator)
        except AttributeError:
            pass
    if total is not None:
        chunksize = int(total / 10)
    else:
        chunksize = 10
    for i, item in enumerate(iterator):
        if (i + 1) % chunksize == 0:
            if total is not None:
                print(f'({i + 1}/{total})')
            else:
                print(f'({i + 1}...)')
        yield item


class Progress:
    style = 'basic'

    def __call__(self, *args, **kwargs):
        if self.style == 'tqdm':
            return tqdm.tqdm(*args, **kwargs)
        elif self.style == 'basic':
            return basic_progress(*args, **kwargs)
        else:
            raise NotImplementedError(f'style of progress not implemented {self.style}')