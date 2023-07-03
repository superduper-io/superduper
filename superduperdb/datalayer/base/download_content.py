# type: ignore
import typing as t

from superduperdb.core.documents import Document
from superduperdb.datalayer.base.query import Insert, Select
from superduperdb.fetchers.downloads import Downloader
from superduperdb.fetchers.downloads import gather_uris
from superduperdb.misc.logger import logging
from superduperdb.queries.serialization import from_dict


def download_content(
    db,
    query: t.Union[Select, Insert, t.Dict],
    ids: t.Optional[t.List[str]] = None,
    documents: t.Optional[t.List[Document]] = None,
    timeout: t.Optional[int] = None,
    raises: bool = True,
    n_download_workers: t.Optional[int] = None,
    headers: t.Optional[t.Dict] = None,
    **kwargs,
):
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
