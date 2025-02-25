import random
import uuid

from superduper import CFG, Listener, Table
from superduper.base.base import Base
from superduper.base.datalayer import Datalayer
from superduper.components.model import Model


class documents(Base):
    id: str
    logical: bool
    number: int
    value: float


def create_documents(db: Datalayer):
    ids = [str(uuid.uuid4()) for _ in range(10)]
    data = [
        documents(
            id=id,
            logical=random.random() < 0.5,
            number=random.randrange(10),
            value=10 * random.random(),
        )
        for id in ids
    ]

    db.insert(data)
    return data, ids


class MyModel(Model):
    def predict(self, x) -> int:
        return x + 1


def test(db):

    ids = [str(uuid.uuid4()) for _ in range(10)]

    _, ids = create_documents(db)

    assert 'documents' in db.databackend.list_tables()

    assert 'documents' in db.show('Table')

    assert db['documents'].get(id=ids[4])['id'] == ids[4]

    pid = db['documents'].primary_id.execute()
    docs = db['documents'].execute()
    pids = [r[pid] for r in docs]

    t = db['documents']

    assert set(t.ids()) == set(pids)

    for r in t.filter(t['number'] > 5).execute():
        assert r['number'] > 5

    for r in db['documents'].select('logical', 'number').execute():
        assert set(r.keys()) == {'logical', 'number'}

    assert len(db['documents'].execute()) == 10
    db['documents'].delete({'id': ids[0]})
    assert len(db['documents'].execute()) == 9

    list = Listener(
        'test',
        model=MyModel('test'),
        select=db['documents'],
        key='number',
    )

    db.apply(list)

    assert db[list.outputs].get() is not None

    results = db['documents'].outputs(list.predict_id).execute()

    for r in results:
        assert r[list.outputs] == r['number'] + 1

    db.databackend.drop_table('documents')

    assert 'documents' not in db.databackend.list_tables()

    db.databackend.drop(force=True)

    assert not db.databackend.list_tables()
