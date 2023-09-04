import typing as t

from superduperdb import CFG, logging
from superduperdb.container.document import Document
from superduperdb.container.serializable import Serializable
from superduperdb.db.base.download import Downloader, SaveFile, gather_uris
from superduperdb.db.base.query import Insert, Select


def download_content(
    db,
    query: t.Union[Select, Insert, t.Dict],
    ids: t.Optional[t.Sequence[str]] = None,
    documents: t.Optional[t.List[Document]] = None,
    timeout: t.Optional[int] = None,
    raises: bool = True,
    n_download_workers: t.Optional[int] = None,
    headers: t.Optional[t.Dict] = None,
    download_update: t.Optional[t.Callable] = None,
    **kwargs,
) -> t.Optional[t.Sequence[Document]]:
    """
    Download content contained in uploaded data. Items to be downloaded are identifier
    via the subdocuments in the form exemplified below. By default items are downloaded
    to the database, unless a ``download_update`` function is provided.

    :param db: database instance
    :param query: query to be executed
    :param ids: ids to be downloaded
    :param documents: documents to be downloaded
    :param timeout: timeout for download
    :param raises: whether to raise errors
    :param n_download_workers: number of download workers
    :param headers: headers to be used for download
    :param download_update: function to be used for updating the database
    :param **kwargs: additional keyword arguments

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
        query = Serializable.deserialize(query)

    if documents is not None:
        pass
    elif isinstance(query, Select):
        update_db = True
        if ids is None:
            documents = list(db.execute(query).raw_cursor)
        else:
            select = query.select_using_ids(ids)
            documents = list(db.execute(select))
    else:
        assert isinstance(query, Insert)
        documents = query.documents

    uris, keys, place_ids = gather_uris([d.encode() for d in documents])
    logging.info(f'found {len(uris)} uris')
    if not uris:
        return None

    if n_download_workers is None:
        try:
            n_download_workers = db.metadata.get_metadata(key='n_download_workers')
        except TypeError:
            n_download_workers = 0

    if headers is None:
        try:
            headers = db.metadata.get_metadata(key='headers')
        except TypeError:
            pass

    if timeout is None:
        try:
            timeout = db.metadata.get_metadata(key='download_timeout')
        except TypeError:
            pass

    if CFG.downloads.hybrid:
        _download_update = SaveFile(CFG.downloads.root)
    else:

        def _download_update(key, id, bytes_, **kwargs):  # type: ignore[misc]
            return query.download_update(db=db, key=key, id=id, bytes=bytes_)

    assert place_ids is not None
    downloader = Downloader(
        uris=uris,
        ids=place_ids,
        keys=keys,
        update_one=download_update or _download_update,
        n_workers=n_download_workers,
        timeout=timeout,
        headers=headers,
        raises=raises,
    )
    downloader.go()
    if update_db:
        return None

    for id_, key in zip(place_ids, keys):
        documents[id_] = db.db.set_content_bytes(  # type: ignore[call-overload]
            documents[id_],  # type: ignore[call-overload]
            key,
            downloader.results[id_],  # type: ignore[index]
        )
    return documents
