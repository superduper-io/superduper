import os
from test.unittest.db.mongodb.test_database import IMAGE_URL

import pytest
import tdir

from superduperdb import CFG
from superduperdb.backends.mongodb.query import Collection
from superduperdb.base.document import Document
from superduperdb.ext.pillow.encoder import pil_image
from superduperdb.misc.download import Fetcher

remote = os.environ.get('SDDB_REMOTE_TEST', 'local')


def test_s3_and_web():
    if remote == 'remote':
        Fetcher()('s3://superduperdb-bucket/img/black.png')


@pytest.fixture
def patch_cfg_downloads(monkeypatch):
    with tdir() as td:
        monkeypatch.setattr(CFG.downloads, 'hybrid', True)
        monkeypatch.setattr(CFG.downloads, 'root', td)
        yield


def test_file_blobs(empty, patch_cfg_downloads):
    to_insert = [
        Document(
            {
                'item': {
                    '_content': {
                        'uri': IMAGE_URL,
                        'encoder': 'pil_image',
                    }
                },
                'other': {
                    'item': {
                        '_content': {
                            'uri': IMAGE_URL,
                            'encoder': 'pil_image',
                        }
                    }
                },
            }
        )
        for _ in range(2)
    ]

    empty.execute(Collection('documents').insert_many(to_insert), encoders=(pil_image,))
    empty.execute(Collection('documents').find_one())
    assert list(CFG.downloads.root.iterdir())
