import os

from superduperdb.db.base.download import Fetcher

remote = os.environ.get('SUPERDUPERDB_REMOTE_TEST', 'local')


def test_s3_and_web():
    if remote == 'remote':
        Fetcher()('s3://superduperdb-bucket/img/black.png')
