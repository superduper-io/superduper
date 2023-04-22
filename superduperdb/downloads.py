from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
import requests
import signal
import sys
import warnings

from superduperdb.progress import progressbar


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


class BaseDownloader:
    def __init__(self, urls, n_workers=0, timeout=None, headers=None, raises=True):
        self.timeout = timeout
        self.n_workers = n_workers
        self.urls = urls
        self.headers = headers or {}
        self.raises = raises

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
                        self._download(i, request_session=request_session)
                else:
                    self._download(i, request_session=request_session)
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


class Downloader(BaseDownloader):
    """

    :param table: table or collection to _download items
    :param urls: list of urls/ file names to fetch
    :param update_one: function to call to insert data into table
    :param ids: list of ids of rows/ documents to update
    :param keys: list of keys in rows/ documents to insert to
    :param n_workers: number of multiprocessing workers
    :param raises: raises error ``True``/``False``
    :param headers: dictionary of request headers passed to``requests`` package
    :param skip_existing: if ``True`` then don't bother getting already present data
    :param timeout: set seconds until request times out
    """
    def __init__(
        self,
        table,
        urls,
        update_one=None,
        ids=None,
        keys=None,
        n_workers=20,
        headers=None,
        skip_existing=True,
        timeout=None,
        raises=True,
    ):
        super().__init__(urls, n_workers=n_workers, timeout=timeout, headers=headers,
                         raises=raises)
        self.table = table
        self.ids = ids
        self.keys = keys
        self.failed = 0
        self.skip_existing = skip_existing
        self.update_one = update_one

        assert len(ids) == len(urls)

    def _download(self, i, request_session):
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

        self.update_one(self.table, self.ids[i], self.keys[i], r.content)


class InMemoryDownloader(BaseDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}

    def _download(self, i, request_session):
        url = self.urls[i]
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

        self.results[i] = r.content


def gather_urls(documents, gather_ids=True):
    """
    Get the URLS out of all documents as denoted by ``{"_content": ...}``

    :param documents: list of dictionaries
    """
    urls = []
    mongo_keys = []
    ids = []
    for i, r in enumerate(documents):
        sub_urls, sub_mongo_keys = _gather_urls_for_document(r)
        if gather_ids:
            ids.extend([r['_id'] for _ in sub_urls])
        else:
            ids.append(i)
        urls.extend(sub_urls)
        mongo_keys.extend(sub_mongo_keys)
    return urls, mongo_keys, ids


def _gather_urls_for_document(r):
    '''
    >>> _gather_urls_for_document({'a': {'_content': {'url': 'test'}}})
    (['test'], ['a'])
    >>> d = {'b': {'a': {'_content': {'url': 'test'}}}}
    >>> _gather_urls_for_document(d)
    (['test'], ['b.a'])
    >>> d = {'b': {'a': {'_content': {'url': 'test', 'bytes': b'abc'}}}}
    >>> _gather_urls_for_document(d)
    ([], [])
    '''
    urls = []
    keys = []
    for k in r:
        if isinstance(r[k], dict) and '_content' in r[k]:
            if 'url' in r[k]['_content'] and 'bytes' not in r[k]['_content']:
                keys.append(k)
                urls.append(r[k]['_content']['url'])
        elif isinstance(r[k], dict) and '_content' not in r[k]:
            sub_urls, sub_keys = _gather_urls_for_document(r[k])
            urls.extend(sub_urls)
            keys.extend([f'{k}.{key}' for key in sub_keys])
    return urls, keys


