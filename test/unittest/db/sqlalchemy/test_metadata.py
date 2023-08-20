import os
import pytest
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


from superduperdb.db.sqlalchemy.metadata import SQLAlchemyMetadata


DATABASE_URL = "sqlite:///.testsqlite.db"


@pytest.fixture
def metadata():
    engine = create_engine(DATABASE_URL)
    store = SQLAlchemyMetadata(
        conn=engine.connect(),
        name='testsqlite'
    )
    yield store
    store.drop(force=True)
    engine.dispose()
    Base.metadata.drop_all(engine)
    os.remove('.testsqlite.db')


def test(metadata):

    metadata.create_component({
        'identifier': 'my-model',
        'type_id': 'model',
        'version': 0,
        'cls': 'Model',
        'module': 'superduperdb.container.model',
    })

    comps = metadata.show_components('model')
    assert comps == ['my-model']

    metadata.create_component({
        'identifier': 'other-model',
        'type_id': 'model',
        'version': 0,
        'cls': 'Model',
        'module': 'superduperdb.container.model',
        'parent': 'model/my-model/0'
    })
    
    comps = metadata.show_components('model')

    r = metadata.get_component(
        type_id='model',
        identifier='other-model',
        version=0,
        allow_hidden=True,
    )

    assert r['id'] == 'model/other-model/0'
