# type: ignore
import typing as t

from superduperdb.core.documents import Document
from superduperdb.datalayer.base.query import Insert, Select
from superduperdb.misc.downloads import Downloader
from superduperdb.misc.downloads import gather_uris
from superduperdb.misc.logger import logging
from superduperdb.misc.serialization import from_dict


def download_content(
    db,
    query: t.Union[Select, Insert, t.Dict],
    ids: t.Optional[t.List[str]] = None,
    documents: t.Optional[t.List[Document]] = None,
    timeout: t.Optional[int] = None,
    raises: bool = True,
    n_download_workers: t.Optional[int] = None,
    headers: t.Optional[t.Dict] = None,
    download_update: t.Optional[t.Callable] = None,
    **kwargs,
):
    """
    Download content contained in uploaded data. Items to be downloaded are identifier
    via the subdocuments in the form exemplified below. By default items are downloaded
    to the database, unless a ``download_update`` function is provided.

    >>> d = {"_content": {"uri": "<uri>", "encoder": "<encoder-identifier>"}}
    >>> def update(key, id, bytes):
    >>> ... with open(f'/tmp/{key}+{id}', 'wb') as f:
    >>> ...     f.write(bytes)
    >>> download_content(None, None, ids=["0"], documents=[d]))
    ...
    """
    logging.debug(query)
    logging.debug(ids)
    update_db = False

    if isinstance(query, dict):
        query = from_dict(query)

    if documents is not None:
        pass
    elif isinstance(query, Select):
        update_db = True
        if ids is None:
            query = query.copy(update={'raw': True})
            documents = list(db.select(query))
        else:
            select = query.select_using_ids(ids)
            select = select.copy(update={'raw': True})
            documents = list(db.select(select))
    else:
        documents = query.documents

    uris, keys, place_ids = gather_uris([d.encode() for d in documents])
    logging.info(f'found {len(uris)} uris')
    if not uris:
        return

    if n_download_workers is None:
        try:
            n_download_workers = db.metadata.get_metadata(key='n_download_workers')
        except TypeError:
            n_download_workers = 0

    if headers is None:
        try:
            headers = db.metadata.get_metadata(key='headers')
        except TypeError:
            headers = 0

    if timeout is None:
        try:
            timeout = db.metadata.get_metadata(key='download_timeout')
        except TypeError:
            timeout = None

    if download_update is None:

        def download_update(key, id, bytes):
            return query.download_update(db=db, key=key, id=id, bytes=bytes)

    downloader = Downloader(
        uris=uris,
        ids=place_ids,
        keys=keys,
        update_one=download_update,
        n_workers=n_download_workers,
        timeout=timeout,
        headers=headers,
        raises=raises,
    )
    downloader.go()
    if update_db:
        return
    for id_, key in zip(place_ids, keys):
        documents[id_] = db.db.set_content_bytes(
            documents[id_], key, downloader.results[id_]
        )
    return documents
