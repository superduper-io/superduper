from __future__ import annotations

import os
import shutil

import pytest

from superduperdb import CFG
from superduperdb.container.document import Document
from superduperdb.db.base.download import Fetcher
from superduperdb.db.mongodb.query import Collection
from superduperdb.ext.pillow.image import pil_image

remote = os.environ.get('SUPERDUPERDB_REMOTE_TEST', 'local')


IMAGE_URL = 'https://www.superduperdb.com/logos/white.png'


def test_s3_and_web():
    if remote == 'remote':
        Fetcher()('s3://superduperdb-bucket/img/black.png')


@pytest.fixture
def make_hybrid_cfg():
    old = CFG.downloads.hybrid
    CFG.downloads.hybrid = True
    root_old = CFG.downloads.root
    CFG.downloads.root = '.tdir_hybrid'
    os.makedirs('.tdir_hybrid', exist_ok=True)
    yield CFG
    CFG.downloads.hybrid = old
    CFG.downloads.root = root_old
    shutil.rmtree('.tdir_hybrid')


def test_file_blobs(empty, make_hybrid_cfg):
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

    empty.execute(Collection('documents').insert_many(to_insert, encoders=(pil_image,)))
    empty.execute(Collection('documents').find_one())
    imgs = os.listdir('.tdir_hybrid/')

    assert imgs
