from superduper.base.base import Base


class New(Base):
    a: str


def test(db):

    db.create(New)

    data = [
        New('testa', a='a').dict(path=False),
        New('testb', a='b').dict(path=False),
        New('testc', a='c').dict(path=False),
    ]

    db['New'].insert(data)

    loaded = db['New'].execute()

    assert len(loaded) == 3

    db['New'].update({'a': 'a'}, 'a', 'a2')

    r = db['New'].get(identifier='testa')

    assert r['a'] == 'a2'

    r['a'] = 'a3'

    db['New'].replace({'identifier': 'testa'}, r)

    assert db['New'].get(identifier='testa')['a'] == 'a3'

    db['New'].delete({'identifier': 'testa'})
    assert len(db['New'].execute()) == 2

    assert db['New'].get(identifier='testa') is None
