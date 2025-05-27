from my_module import my_func, my_other_func

from superduper import ObjectModel, superduper
from superduper.base.event import Job


def test():

    db = superduper(cluster_engine='simple')

    m = ObjectModel('test', object=my_func)
    n = ObjectModel('other', object=my_other_func)

    job1 = Job(
        context='testingtesting123',
        component='ObjectModel',
        identifier='test',
        uuid=m.uuid,
        args=(1,),
        method='predict',
        kwargs={},
    )

    db.apply(m, force=True)
    db.apply(n, force=True)

    job2 = Job(
        context='testingtesting123',
        component='ObjectModel',
        identifier='other',
        uuid=n.uuid,
        args=(2,),
        method='predict',
        kwargs={},
    )

    job1.execute(db)
    job2.execute(db)

    job1.wait(db)
    job2.wait(db)

    jobs = db['Job'].execute()

    for j in jobs:
        msg = f"Job {j['uuid']} failed with status {j['status']}"
        assert j['status'] == 'success', msg
