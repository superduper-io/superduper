from superduper.base.document import Document


def test(db):
    q = db['docs'].select()

    assert hasattr(q, 'filter')

    ###

    q = db['docs'].select('a').select('b')

    assert str(q) == 'docs.select("a", "b")'

    ###

    t = db['docs']
    q = t.select('a').filter(t['a'] == 2)

    assert str(q) == 'docs.filter(docs["a"] == 2).select("a")'


def test_decomposition(db):
    t = db['docs']

    q = t.filter(t['a'] == 2).select('a')

    d = q.decomposition

    assert d.filter is not None and d.select is not None

    assert str(d.to_query()) == str(q)


def test_stringify(db):
    t = db['docs']

    q = t.filter(t['a'] == 2).select('a')

    assert str(q) == 'docs.filter(docs["a"] == 2).select("a")'

    q = t.like({'a': 2}, vector_index='test').select()

    assert str(q) == "docs.like({'a': 2}, \"test\", n=10).select()"


def test_serialize_deserialize(db):
    t = db['docs']

    q = t.like({'a': 2}, vector_index='test').select()

    r = q.dict()

    de_q = Document.decode(r, db=db).unpack()

    assert str(q) == str(de_q)
