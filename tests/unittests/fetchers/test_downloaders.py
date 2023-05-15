from superduperdb.fetchers.downloads import Fetcher
import os
from tests.material.types import Image


remote = os.environ.get('SUPERDUPERDB_REMOTE_TEST', 'local')


def test_s3_and_web():
    if remote == 'remote':
        img = Fetcher()('s3://superduperdb-bucket/img/black.png')
        print(Image.decode(img))
