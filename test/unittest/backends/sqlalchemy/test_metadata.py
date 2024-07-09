import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

from superduper.backends.sqlalchemy.metadata import SQLAlchemyMetadata

Base = declarative_base()

DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture
def metadata():
    engine = create_engine(DATABASE_URL)
    store = SQLAlchemyMetadata(DATABASE_URL)
    yield store
    store.drop(force=True)
    engine.dispose()
    Base.metadata.drop_all(engine)


def test(metadata):
    import uuid

    metadata.create_component(
        {
            'identifier': 'my-model',
            'type_id': 'model',
            'version': 0,
            '_path': 'superduper.container.model.Model',
            'uuid': str(uuid.uuid4()),
        }
    )

    comps = metadata.show_components('model')
    assert comps == ['my-model']

    metadata.create_component(
        {
            'identifier': 'other-model',
            'type_id': 'model',
            'version': 0,
            '_path': 'superduper.container.model.Model',
            'uuid': str(uuid.uuid4()),
        }
    )

    comps = metadata.show_components('model')

    r = metadata.get_component(
        type_id='model',
        identifier='other-model',
        version=0,
        allow_hidden=True,
    )

    assert r['identifier'] == 'other-model'
    assert r['version'] == 0
