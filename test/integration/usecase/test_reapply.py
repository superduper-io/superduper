from superduper import Model


class MyModel(Model):
    breaks = ('b',)
    signature: str = 'singleton'
    a: str
    b: str

    def predict(self, x):
        return x + 1


def test_reapply(db):
    db.cfg.auto_schema = True

    db['docs'].insert([{'x': i} for i in range(10)]).execute()

    def build(name, data):
        model = MyModel('test', a=name, b=data)
        listener_1 = model.to_listener(
            key='x',
            select=db['docs'].select(),
            identifier='listener_1',
        )
        return model.to_listener(
            key=listener_1.outputs,
            select=db[listener_1.outputs].select(),
            identifier='listener_2',
            upstream=[listener_1],
        )

    listener_2 = build('first', '1')

    db.apply(listener_2)

    outputs = db[listener_2.outputs].select().tolist()

    import pprint

    pprint.pprint(outputs)

    assert outputs[0][listener_2.outputs] == 2

    listener_2 = build('second', '1')
    db.apply(listener_2)

    listener_2_update = build('second', '2')

    db.apply(listener_2_update)
