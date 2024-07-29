from test.utils.setup.fake_data import add_random_data

from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel
from superduper.components.template import QueryTemplate, Template


def test_basic_template(db):
    db.cfg.auto_schema = True
    add_random_data(db)

    def model(x):
        return x + 2

    m = Listener(
        model=ObjectModel(
            object=model,
            identifier='<var:model_id>',
        ),
        select=db['documents'].select(),
        key='<var:key>',
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    template = Template(
        identifier='my-template',
        template=m.encode(metadata=False),
    )

    vars = template.template_variables
    assert len(vars) == 2
    assert all([v in ['key', 'model_id'] for v in vars])
    db.apply(template)

    # Check template component has not been added to metadata
    assert 'my_id' not in db.show('model')
    assert all([ltr.split('/')[-1] != m.identifier for ltr in db.show('listener')])
    listener = template(key='y', model_id='my_id')

    assert listener.key == 'y'
    assert listener.model.identifier == 'my_id'

    db.apply(listener)

    reloaded_template = db.load('template', template.identifier)
    listener = reloaded_template(key='y', model_id='my_id')

    db.apply(listener)

    listener.init()
    assert listener.model.object(3) == 5

    # Check listener outputs with key and model_id
    r = db['documents'].select().outputs(listener.predict_id).execute()
    r = Document(list(r)[0].unpack())
    assert r[listener.outputs_key] == r['y'] + 2


def test_template_export(db):
    db.cfg.auto_schema = True
    add_random_data(db)
    m = Listener(
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier='<var:model_id>',
        ),
        select=db['<var:collection>'].select(),
        key='<var:key>',
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    t = Template(
        identifier='my-template',
        template=m.encode(),
    )

    db.apply(t)

    t = db.load('template', t.identifier)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        t.export(temp_dir)

        rt = Component.read(temp_dir, db=db)
        db.apply(rt)

        listener = rt(key='y', model_id='my_id', collection='documents')

        assert listener.key == 'y'
        assert listener.model.identifier == 'my_id'
        assert listener.select.table == 'documents'

        db.apply(listener)
        # Check listener outputs with key and model_id
        r = db['documents'].select().outputs(listener.predict_id).execute()
        r = Document(list(r)[0].unpack())
        assert r[listener.outputs_key] == r['y'] + 2


def test_from_template(db):
    add_random_data(db)
    m = Listener(
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier='<var:model_id>',
        ),
        select=db['<var:collection>'].select(),
        key='<var:key>',
    )
    component = Component.from_template(
        identifier='test-from-template',
        template_body=m.encode(),
        key='y',
        model='my_id',
    )

    component.init()
    assert isinstance(component, Listener)
    assert isinstance(component.model, ObjectModel)
    assert component.model.object(3) == 5


def test_query_template(db):
    add_random_data(db)
    q = db['documents'].find({'this': 'is a <var:test>'}).limit('<var:limit>')
    t = QueryTemplate('select_lim', template=q)

    assert set(t.template_variables) == {'limit', 'test'}
    assert t.template['query'] == 'documents.find(documents[0]).limit("<var:limit>")'
