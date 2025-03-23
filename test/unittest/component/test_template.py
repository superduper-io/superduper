from test.utils.setup.fake_data import add_random_data

from superduper.base.document import Document
from superduper.components.component import Component
from superduper.components.listener import Listener
from superduper.components.model import ObjectModel
from superduper.components.template import Template


def test_basic_template(db):
    add_random_data(db)

    def model(x):
        return x + 2

    m = Listener(
        identifier='lm',
        model=ObjectModel(
            object=model,
            identifier='test',
            datatype='int',
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
    assert len(vars) == 1
    assert all([v in ['key', 'model_id'] for v in vars])
    db.apply(template)

    # Check template component has not been added to metadata
    assert 'my_id' not in db.show('ObjectModel')

    assert all([ltr.split('/')[-1] != m.identifier for ltr in db.show('Listener')])

    listener = template(key='y')

    assert listener.key == 'y'

    db.apply(listener)

    reloaded_template = db.load('Template', template.identifier)
    listener = reloaded_template(key='y')

    db.apply(listener)

    listener.setup()
    assert listener.model.object(3) == 5

    # Check listener outputs with key and model_id
    r = db['documents'].outputs(listener.predict_id).execute()
    r = Document(list(r)[0].unpack())
    assert r[listener.outputs] == r['y'] + 2


def test_template_export(db):
    add_random_data(db)
    m = Listener(
        identifier='lm',
        model=ObjectModel(
            object=lambda x: x + 2,
            identifier='test',
            datatype='int',
        ),
        select=db['<var:table>'].select(),
        key='<var:key>',
    )

    # Optional "info" parameter provides details about usage
    # (depends on developer use-case)
    t = Template(
        identifier='my-template',
        template=m.encode(),
    )

    db.apply(t)

    t = db.load('Template', t.identifier)

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        t.export(temp_dir)

        rt = Component.read(temp_dir)
        db.apply(rt)

        listener = rt(key='y', table='documents')

        assert listener.key == 'y'
        assert listener.select.table == 'documents'

        db.apply(listener)
        # Check listener outputs with key and model_id
        r = db['documents'].outputs(listener.predict_id).execute()
        r = Document(list(r)[0].unpack())
        assert r[listener.outputs] == r['y'] + 2


def test_cross_reference(db):
    from superduper import Application

    m = ObjectModel(
        object=lambda x: x + 2,
        identifier='my_id',
    )

    l1 = Listener(model=m, select=db['docs'].select(), key='x', identifier='l1')
    l2 = Listener(
        model=m, select=db[l1.outputs].select(), key=l1.outputs, identifier='l2'
    )

    app = Application('my-app', components=[l1, l2])

    r = app.encode(metadata=False, defaults=False)

    r.pop('_blobs')

    import pprint

    pprint.pprint(r)
