from superduperdb.fetchers.downloads import Fetcher
from tests.material.types import Image


def test_s3_and_web():
    img = Fetcher()('s3://superduperdb-bucket/img/black.png')
    print(Image.decode(img))
