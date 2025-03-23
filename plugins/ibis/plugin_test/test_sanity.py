from superduper.base.base import Base


class New(Base):
    a: str


def test(db):

    data = [New(a='a'), New(a='b'), New(a='c')]

    db.insert(data)

    loaded = db['New'].execute()

    assert len(loaded) == 3

    db['New'].update({'a': 'a'}, 'a', 'a2')

    r = db['New'].get(a='a2')

    assert r['a'] == 'a2'

    r['a'] = 'a3'

    db['New'].replace({'a': 'a2'}, r)

    assert db['New'].get(a='a3')['a'] == 'a3'

    db['New'].delete({'a': 'a3'})
    assert len(db['New'].execute()) == 2

    assert db['New'].get(a='a3') is None
