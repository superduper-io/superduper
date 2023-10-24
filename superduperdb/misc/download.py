import hashlib
import os
import re
import signal
import sys
import typing as t
import warnings
from contextlib import contextmanager
from io import BytesIO
from multiprocessing.pool import ThreadPool

import boto3
import requests
from tqdm import tqdm

from superduperdb import CFG, logging


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


class Fetcher:
    """
    Fetches data from a URI

    :param headers: headers to be used for download
    :param n_workers: number of download workers
    """

    def __init__(self, headers: t.Optional[t.Dict] = None, n_workers: int = 0):
        session = boto3.Session()
        self.headers = headers
        self.s3_client = session.client("s3")
        self.request_session = requests.Session()
        self.request_adapter = requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=n_workers if n_workers else 1,
            pool_maxsize=n_workers * 10,
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
        path = re.split('^file://', path)[-1]
        with open(path, 'rb') as f:
            return f.read()

    def _download_from_uri(self, uri):
        return self.request_session.get(uri, headers=self.headers).content

    def __call__(self, uri: str):
        """
        Download data from a URI

        :param uri: uri to download from
        """
        if uri.startswith('file://'):
            return self._download_file(uri)
        elif uri.startswith('s3://'):
            return self._download_s3_object(uri)
        elif uri.startswith('http://') or uri.startswith('https://'):
            return self._download_from_uri(uri)
        else:
            raise NotImplementedError(f'unknown type of URI "{uri}"')


class BaseDownloader:
    """
    Base class for downloading files

    :param uris: list of uris/ file names to fetch
    :param n_workers: number of multiprocessing workers
    :param timeout: set seconds until request times out
    :param headers: dictionary of request headers passed to``requests`` package
    :param raises: raises error ``True``/``False``
    """

    def __init__(
        self,
        uris: t.List[str],
        n_workers: int = 0,
        timeout: t.Optional[int] = None,
        headers: t.Optional[t.Dict] = None,
        raises: bool = True,
    ):
        self.timeout = timeout
        self.n_workers = n_workers
        self.uris = uris
        self.headers = headers or {}
        self.raises = raises

    def go(self):
        """
        Download all files
        Uses a :py:class:`multiprocessing.pool.ThreadPool` to parallelize
                          connections.
        """
        logging.info(f'number of workers {self.n_workers}')
        prog = tqdm(total=len(self.uris))
        prog.prefix = 'downloading from uris'
        self.failed = 0
        prog.prefx = "failed: 0"

        def f(i):
            prog.update()
            try:
                if self.timeout is not None:
                    with timeout(self.timeout):
                        self._download(i)
                else:
                    self._download(i)
            except TimeoutException:
                logging.warning(f'timed out {i}')
            except Exception as e:
                if self.raises:
                    raise e
                warnings.warn(str(e))
                self.failed += 1
                prog.prefix = f"failed: {self.failed} [{e}]"

        if self.n_workers == 0:
            self._sequential_go(f)
            return

        self._parallel_go(f)

    def _check_exists_if_hybrid(self, uri):
        if uri.startswith('file://'):
            file = f'{CFG.downloads.root}/{uri.split("file://")[-1]}'
        else:
            file = f'{CFG.downloads.root}/{hashlib.sha1(uri.encode()).hexdigest()}'
        if os.path.exists(file):
            return True
        return False

    def _parallel_go(self, f):
        pool = ThreadPool(self.n_workers)
        try:
            pool.map(f, range(len(self.uris)))
        except KeyboardInterrupt:
            logging.warning("--keyboard interrupt--")
            pool.terminate()
            pool.join()
            sys.exit(1)  # Kill this subprocess so it doesn't hang

        pool.close()
        pool.join()

    def _sequential_go(self, f):
        for i in range(len(self.uris)):
            f(i)


class SaveFile:
    """
    Save file to disk.

    :param root: root directory to save files to.
    """

    def __init__(self, root: str):
        self.root = root

    def __call__(self, bytes_: bytearray, uri: str, **kwargs) -> None:
        path = f'{self.root}/{hashlib.sha1(uri.encode()).hexdigest()}'
        with open(path, 'wb') as f:
            f.write(bytes_)


class Downloader(BaseDownloader):
    """
    Download files from a list of URIs.

    :param uris: list of uris/ file names to fetch
    :param update_one: function to call to insert data into table
    :param ids: list of ids of rows/ documents to update
    :param keys: list of keys in rows/ documents to insert to
    :param n_workers: number of multiprocessing workers
    :param headers: dictionary of request headers passed to``requests`` package
    :param skip_existing: if ``True`` then don't bother getting already present data
    :param timeout: set seconds until request times out
    :param raises: raises error ``True``/``False``
    """

    results: t.Dict[int, str]

    def __init__(
        self,
        uris,
        update_one: t.Optional[t.Callable] = None,
        ids: t.Optional[t.Union[t.List[str], t.List[int]]] = None,
        keys: t.Optional[t.List[str]] = None,
        n_workers: int = 20,
        headers: t.Optional[t.Dict] = None,
        skip_existing: bool = True,
        timeout: t.Optional[int] = None,
        raises: bool = True,
    ):
        super().__init__(
            uris, n_workers=n_workers, timeout=timeout, headers=headers, raises=raises
        )

        if ids is not None:
            if len(ids) != len(uris):
                raise ValueError(f'len(ids={ids}) != len(uris={uris})')

        self.ids = ids
        self.keys = keys
        self.failed = 0
        self.skip_existing = skip_existing
        self.update_one = update_one
        self.fetcher = Fetcher(headers=headers, n_workers=n_workers)

    def _download(self, i):
        if CFG.downloads.hybrid:
            if self._check_exists_if_hybrid(self.uris[i]):
                return
        content = self.fetcher(self.uris[i])
        self.update_one(
            id=self.ids[i],
            key=self.keys[i],
            bytes_=content,
            uri=self.uris[i],
        )


def gather_uris(
    documents: t.Sequence[t.Dict], gather_ids: bool = True
) -> t.Tuple[t.List[str], t.List[str], t.Union[t.List[int], t.List[str]]]:
    """
    Get the uris out of all documents as denoted by ``{"_content": ...}``

    :param documents: list of dictionaries
    :param gather_ids: if ``True`` then gather ids of documents
    """
    uris = []
    mongo_keys = []
    ids = []
    for i, r in enumerate(documents):
        sub_uris, sub_mongo_keys = _gather_uris_for_document(r)
        if gather_ids:
            ids.extend([r['_id'] for _ in sub_uris])
        else:
            ids.append(i)
        uris.extend(sub_uris)
        mongo_keys.extend(sub_mongo_keys)
    return uris, mongo_keys, ids


def _gather_uris_for_document(r: t.Dict):
    '''
    >>> _gather_uris_for_document({'a': {'_content': {'uri': 'test'}}})
    (['test'], ['a'])
    >>> d = {'b': {'a': {'_content': {'uri': 'test'}}}}
    >>> _gather_uris_for_document(d)
    (['test'], ['b.a'])
    >>> d = {'b': {'a': {'_content': {'uri': 'test', 'bytes': b'abc'}}}}
    >>> _gather_uris_for_document(d)
    ([], [])
    '''
    uris = []
    keys = []
    for k in r:
        if isinstance(r[k], dict) and '_content' in r[k]:
            if 'uri' in r[k]['_content'] and 'bytes' not in r[k]['_content']:
                keys.append(k)
                uris.append(r[k]['_content']['uri'])
        elif isinstance(r[k], dict) and '_content' not in r[k]:
            sub_uris, sub_keys = _gather_uris_for_document(r[k])
            uris.extend(sub_uris)
            keys.extend([f'{k}.{key}' for key in sub_keys])
    return uris, keys
