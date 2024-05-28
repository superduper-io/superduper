import pytest

from superduperdb.base.document import Document
from superduperdb.components.application import Application
from test.db_config import DBConfig

from superduperdb.base.variables import Variable
from superduperdb.components.model import ObjectModel
from superduperdb.components.template import Template
from superduperdb import Listener


@pytest.mark.parametrize("db", [DBConfig.mongodb_empty], indirect=True)
def test_template(db):


    model = ObjectModel(
        identifier=Variable('name'),
        object=lambda x: x + 2,
    )

    m = Listener(
        identifier='test',
        model=model,
        key=Variable('key'),
        select=db['collection'].find()
    )

    template = Template(
        'model_template',
        template=m,
        info={'name': {'type': 'str'}, 'key': {'type': 'str'}},
    )

    db.apply(template)

    reloaded = db.load('template', template.identifier)
    component = reloaded(name='my_name', key='X') 

    application = Application(
        identifier='my_application',
        template=template,
        kwargs={'name': 'my_name', 'key': 'X'},
    )

    db.apply(application.copy())


@pytest.mark.parametrize("db", [DBConfig.mongodb], indirect=True)
def test_variables_on_query(db):
    q = db[Variable('collection')].find({'test': Variable('test')})

    r = q.encode()

    assert r['_leaves'][r['_base'][1:]]['query'] == '?collection.find(documents[0])'

    qq = Document.decode(r).unpack()

    print(qq)

    print(qq.variables)

    assert str(qq.variables) == '[?test, ?collection]'

    qqq = qq.set_variables(collection='docs', test='my_test')

    print(qqq)

    assert not qqq.variables
