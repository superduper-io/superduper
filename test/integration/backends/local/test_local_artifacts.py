import PIL.Image
import PIL.PngImagePlugin
import pytest

from superduperdb import CFG
from superduperdb.backends.mongodb import Collection
from superduperdb.base.document import Document

DO_SKIP = not CFG.data_backend.startswith("mongodb")


@pytest.fixture
def image(test_db):
    img = PIL.Image.open('test/material/data/test.png')
    from superduperdb.ext.pillow.encoder import image_type

    _, i = test_db.add(image_type('image'))

    insert = Collection('images').insert_one(Document({'img': i(img)}))

    test_db.execute(insert)

    yield test_db

    img.close()

    test_db.databackend.conn.test_db.drop_collection('images')


@pytest.mark.skipif(DO_SKIP, reason="skipping test if not mongodb")
def test_save_artifact(image):
    r = image.execute(Collection('images').find_one())

    r = r.unpack(db=image)

    assert isinstance(r['img'], PIL.PngImagePlugin.PngImageFile)
