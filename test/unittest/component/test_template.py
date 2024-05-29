from test.db_config import DBConfig

import pytest

from superduperdb.base.variables import Variable
from superduperdb.components.listener import Listener
from superduperdb.components.model import ObjectModel
from superduperdb.components.template import Template


@pytest.mark.parametrize('db', [DBConfig.mongodb], indirect=True)
def test_basic_template(db):
    def model(x):
        return x + 2

    m = Listener(
        model=ObjectModel(
            object=model,
            identifier=Variable('model_id'),
        ),
        select=db['documents'].find(),
        key=Variable('key'),
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    template = Template(
        identifier='my-template',
        component=m,
        info={'key': {'type': 'str'}, 'model_id': {'type': 'str'}},
    )
    vars = template.variables
    assert len(vars) == 2
    assert all([v in ['key', 'model_id'] for v in vars])
    db.apply(template)

    # Check template component is not been added to metadata
    assert 'my_id' not in db.show('model')
    assert all([ltr.split('/')[-1] != m.identifier for ltr in db.show('listener')])

    listener = template(key='y', model_id='my_id')

    assert listener.key == 'y'
    assert listener.model.identifier == 'my_id'

    db.apply(listener)
    reloaded_template = db.load('template', template.identifier)
    listener = reloaded_template(key='y', model_id='my_id')

    assert listener.key == 'y'
    assert listener.model.identifier == 'my_id'

    # Check listener outputs with key and model_id
    r = db['documents'].find_one().execute()
    assert r[listener.outputs_key] == r['y'] + 2

    # Try to encode the reloaded_template
    reloaded_data = reloaded_template.encode()
    from superduperdb import Document

    reloaded_template = Document.decode(reloaded_data).unpack()
    listener = reloaded_template(key='y', model_id='my_id')

    assert listener.key == 'y'
    assert listener.model.identifier == 'my_id'
    assert listener.model.object(1) == model(1)


@pytest.mark.parametrize('db', [DBConfig.mongodb], indirect=True)
def test_basic_application(db):
    m = Listener(
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier=Variable('model_id'),
        ),
        select=db['documents'].find(),
        key=Variable('key'),
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    t = Template(
        identifier='my-template',
        component=m,
        info={'key': {'type': 'str'}, 'model_id': {'type': 'str'}},
    )

    db.apply(t)

    from superduperdb.components.application import Application

    application = Application(
        'my_app',
        template=t.identifier,
        kwargs={
            'model_id': 'my_model_id',
            'key': 'y',
        },
    )

    # This loads the template, applies the variables and then
    # applies the built component
    db.apply(application)

    # Check listener outputs with key and model_id
    r = db['documents'].find_one().execute()

    assert '_outputs' in r


@pytest.mark.parametrize('db', [DBConfig.mongodb], indirect=True)
def test_complex_template(db):
    m = Listener(
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier=Variable('model_id'),
        ),
        select=db[Variable('collection')].find(),
        key=Variable('key'),
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    t = Template(
        identifier='my-template',
        component=m,
        info={
            'key': {'type': 'str'},
            'model_id': {'type': 'str'},
            'collection': {'type': 'str'},
        },
    )

    db.apply(t)
    # Check template component is not been added to metadata

    assert 'my_id' not in db.show('model')
    assert all([ltr.split('/')[-1] != m.identifier for ltr in db.show('listener')])

    listener = t(key='y', model_id='my_id', collection='documents')

    assert listener.key == 'y'
    assert listener.model.identifier == 'my_id'
    assert listener.select.identifier == 'documents'

    db.apply(listener)

    # Check listener outputs with key and model_id
    r = db['documents'].find_one().execute()
    assert r[listener.outputs_key] == r['y'] + 2


@pytest.mark.parametrize('db', [DBConfig.mongodb], indirect=True)
def test_template_export(db):
    m = Listener(
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier=Variable('model_id'),
        ),
        select=db[Variable('collection')].find(),
        key=Variable('key'),
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    t = Template(
        identifier='my-template',
        component=m,
        info={
            'key': {'type': 'str'},
            'model_id': {'type': 'str'},
            'collection': {'type': 'str'},
        },
    )

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        t.export(temp_dir)
        rt = Template.read(temp_dir)
        listener = rt(key='y', model_id='my_id', collection='documents')
        assert listener.key == 'y'
        assert listener.model.identifier == 'my_id'
        assert listener.select.identifier == 'documents'

        db.apply(listener)
        # Check listener outputs with key and model_id
        r = db['documents'].find_one().execute()
        assert r[listener.outputs_key] == r['y'] + 2
