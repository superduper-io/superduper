import boto3
from io import BytesIO
from contextlib import contextmanager
from multiprocessing.pool import ThreadPool
import requests
import signal
import sys
import warnings

from superduperdb.misc.progress import progressbar


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


class Fetcher:
    def __init__(self, headers, n_workers):
        session = boto3.Session()
        self.headers = headers
        self.s3_client = session.client("s3")
        self.request_session = requests.Session()
        self.request_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=n_workers if n_workers else 1,
            pool_maxsize=n_workers * 10
        )
        self.request_session.mount("http://", self.request_adapter)
        self.request_session.mount("https://", self.request_adapter)

    def _download_s3_object(self, uri):
        f = BytesIO()
        path = uri.split('s3://')[-1]
        bucket_name = path.split('/')[0]
        file = '/'.join(path.split('/')[1:])
        self.s3_client.download_fileobj(bucket_name, file, f)
        return f.getvalue()

    def _download_file(self, path):
        with open(path, 'rb') as f:
            return f.read()

    def _download_from_url(self, url):
        return self.request_session.get(url, headers=self.headers).content

    def __call__(self, uri):
        if uri.startswith('file://'):
            return self._download_file(uri)
        elif uri.startswith('s3://'):
            return self._download_s3_object(uri)
        elif uri.startswith('http://') or uri.startswith('https://'):
            return self._download_from_url(uri)
        else:
            raise NotImplementedError(f'unknown type of URI "{uri}"')


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

        def f(i):
            prog.update()
            try:
                if self.timeout is not None:  # pragma: no cover
                    with timeout(self.timeout):
                        self._download(i)
                else:
                    self._download(i)
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
        self.fetcher = Fetcher(headers=headers, n_workers=n_workers)

        assert len(ids) == len(urls)

    def _download(self, i):
        url = self.urls[i]
        _id = self.ids[i]
        content = self.fetcher(url)
        self.update_one(self.table, self.ids[i], self.keys[i], content)


class InMemoryDownloader(BaseDownloader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.results = {}

    def _download(self, i, request_session):
        url = self.urls[i]
        if url.startswith('http://') or url.startswith('https://'):
            r = request_session.get(url, headers=self.headers)
        elif url.startswith('s3://'):
            ...
        elif url.startswith('file://'):
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


