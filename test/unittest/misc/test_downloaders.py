import os
import tempfile

import pytest

from superduper import CFG
from superduper.backends.mongodb.query import MongoQuery
from superduper.base.document import Document
from superduper.ext.pillow.encoder import pil_image
from superduper.misc.download import Fetcher

remote = os.environ.get('SDDB_REMOTE_TEST', 'local')


def test_s3_and_web():
    if remote == 'remote':
        Fetcher()('s3://superduper-bucket/img/black.png')


@pytest.fixture
def patch_cfg_downloads(monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setattr(CFG.downloads, 'folder', td)
        yield


# TODO: use table to test the sqldb
@pytest.mark.skipif(True, reason='URI not working')
def test_file_blobs(db, patch_cfg_downloads, image_url):
    db.apply(pil_image)
    to_insert = [Document({"item": pil_image(uri=image_url)}) for _ in range(2)]

    db.execute(MongoQuery(table='documents').insert_many(to_insert))
    r = db.execute(MongoQuery(table='documents').find_one())

    import PIL.PngImagePlugin

    assert isinstance(r.unpack()['item'], PIL.PngImagePlugin.PngImageFile)
